# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
Original modeling. CPU intensive task. But accuracy is good enough tested by preliminary experiments.
'''
import ast
from enum import Flag
import json
import os
import random
import statistics
from abc import ABC
from collections import defaultdict

from this import s
from typing import List, Dict

import numpy as np
from sklearn import ensemble
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

from pet.utils import InputExample, exact_match, save_logits, save_predictions, LogitsList, set_seed, calculate_mean_max_logits
from pet.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
from pet.ipet import *

import transformers
transformers.logging.set_verbosity_error()

import gc

from fed.model import *
from fed.augment import *
from fed.utils import *
from fed.bitfit import *
from fed.selective_annotating import *

import logging
process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

debug = False
eval_step = 1
merge_eval = True
correct_label = False
check_eval = False
staleness = True # whether train on the same clients inferred in this round. Staleness paves the way for asynchronous/pipeline
vote_k_check = False
random_filter = False
select = 'vote' # 'vote' or 'sort'
bitfit_training = True
augment = 'fixed' # 'curriculum' or 'fixed'
# conver_point = 10
# aug_data_point = 100
# vanilla = False # whether fed vanilla is on, fed vanilla means no augmentation, but is fed, means using ft instead of pl to train local model, and aggregate the model via fedavg
# aggregated = True # 是否将10个client训练出来的模型fedavg一下，只infer一次;后面的两个函数里也要改一下
# augmentation = False # 如果不开augmentation，那每个client只能依靠自己的数据来进行训练，而且也不会用到unlabeled data (origin), fed is off
# fed = False

class PetConfig(ABC):
    """Abstract class for a PET configuration that can be saved to and loaded from a json file."""

    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, 'w', encoding='utf8') as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, 'r', encoding='utf8') as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    """Configuration for training a model."""

    def __init__(self, device: str = None, per_gpu_train_batch_size: int = 8, per_gpu_unlabeled_batch_size: int = 8,
                 n_gpu: int = 1, num_train_epochs: int = 3, max_steps: int = -1, gradient_accumulation_steps: int = 1,
                 weight_decay: float = 0.0, learning_rate: float = 5e-5, adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0, max_grad_norm: float = 1, lm_training: bool = False, use_logits: bool = False,
                 alpha: float = 0.9999, temperature: float = 1):
        """
        Create a new training config.

        :param device: the device to use ('cpu' or 'gpu')
        :param per_gpu_train_batch_size: the number of labeled training examples per batch and gpu
        :param per_gpu_unlabeled_batch_size: the number of unlabeled examples per batch and gpu
        :param n_gpu: the number of gpus to use
        :param num_train_epochs: the number of epochs to train for
        :param max_steps: the maximum number of steps to train for (overrides ``num_train_epochs``)
        :param gradient_accumulation_steps: the number of steps to accumulate gradients for before performing an update
        :param weight_decay: the weight decay to use
        :param learning_rate: the maximum learning rate to use
        :param adam_epsilon: the epsilon value for Adam
        :param warmup_steps: the number of warmup steps to perform before reaching the maximum learning rate
        :param max_grad_norm: the maximum norm for the gradient
        :param lm_training: whether to perform auxiliary language modeling (only for MLMs)
        :param use_logits: whether to use each training example's logits instead of its label (used for distillation)
        :param alpha: the alpha parameter for auxiliary language modeling
        :param temperature: the temperature for distillation
        """
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature


class EvalConfig(PetConfig):
    """Configuration for evaluating a model."""

    def __init__(self, device: str = None, n_gpu: int = 1, per_gpu_eval_batch_size: int = 8,
                 metrics: List[str] = None, decoding_strategy: str = 'default', priming: bool = False):
        """
        Create a new evaluation config.

        :param device: the device to use ('cpu' or 'gpu')
        :param n_gpu: the number of gpus to use
        :param per_gpu_eval_batch_size: the number of evaluation examples per batch and gpu
        :param metrics: the evaluation metrics to use (default: accuracy only)
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        :param priming: whether to use priming
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy
        self.priming = priming


class IPetConfig(PetConfig):
    """Configuration for iterative PET training."""

    def __init__(self, generations: int = 3, logits_percentage: float = 0.25, scale_factor: float = 5,
                 n_most_likely: int = -1):
        """
        Create a new iPET config.

        :param generations: the number of generations to train
        :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
        :param scale_factor: the factor by which the training set is increased for each generation
        :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
        """
        self.generations = generations
        self.logits_percentage = logits_percentage
        self.scale_factor = scale_factor
        self.n_most_likely = n_most_likely


def init_model(config: WrapperConfig) -> TransformerModelWrapper:
    """Initialize a new model from the given config."""
    assert config.pattern_id is not None, 'A pattern_id must be set for initializing a new PET model'
    model = TransformerModelWrapper(config)
    return model


def train_fedpet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
               ensemble_eval_config: EvalConfig, ipet_config: IPetConfig, final_model_config: WrapperConfig,
               final_train_config: TrainConfig, final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str,
               ensemble_repetitions: int = 3, final_repetitions: int = 1, reduction: str = 'wmean',
               train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
               eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True, seed: int = 42, aggregated: bool = True,
               augmentation: bool = True, fed: bool = True, vanilla: bool = True, beta: int = None, client_num_in_total: int = None, check_data: List[InputExample] = None, all_client_num_in_total: int = None, labeled_idx: List[int] = None, aug_data_point: int = 100, conver_point: int = 0, limit: int = 0, num_clients_infer: int = 5, infer_freq: int = 1, args = None):
    """
    Train and evaluate a new fed PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
    :param ipet_config: the iPET training configuration
    :param final_model_config: the model configuration for the final distilled sequence classifier
    :param final_train_config: the training configuration for the final distilled sequence classifier
    :param final_eval_config: the evaluation configuration for the final distilled sequence classifier
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ensemble_repetitions: the number of training repetitions for each model corresponding to an individual PVP
    :param final_repetitions: the number of training repetitions for the final distilled sequence classifier
    :param reduction: the reduction strategy for merging predictions, either 'mean' or 'wmean'
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_data_all = train_data
    # logging.info(train_data_all)
    unlabeled_data_all = unlabeled_data
    eval_data_all = eval_data
    
    train_data_sperate, unlabeled_data_seperate, eval_data_seperate = find_labeled(labeled_idx, train_data_all, unlabeled_data_all, eval_data_all)

    # Calculate the samples numbers of all clients involved in training
    sample_num_list = np.array([])
    for client in range(client_num_in_total):
        sample_num_list = np.append(sample_num_list, len(train_data_sperate[client]))
    logging.info("All clients: sample_num_list is {}".format(sample_num_list))

    for gen in range(ipet_config.generations): # debug mode: start from 2nd iteration; defalut is 0
        delete_cache(gen, output_dir)
        logging.info(f"Start generation {gen}.")
        # get the sample num list of the last iteration for fedavg aggregation
        # those train_data_seperate will be different in the second round, and thus leading to null error
        

        # Data augmentation
        sample_num_list = []
        infer_sample_num_list = []
        if augmentation:
            if gen > 0:
                # client selection
                train_data_sperate, unlabeled_data_seperate, eval_data_seperate, curr_sample_num_list, client_indexes, num_clients = client_selection(gen, augmentation, train_data_all, unlabeled_data_all, eval_data_all, train_data_sperate, unlabeled_data_seperate, eval_data_seperate, all_client_num_in_total, client_num_in_total, labeled_idx, conver_point, num_clients_infer)

            else: 
                train_data_sperate, unlabeled_data_seperate, eval_data_seperate, sample_num_list, client_indexes, num_clients = client_selection(gen, augmentation, train_data_all, unlabeled_data_all, eval_data_all, train_data_sperate, unlabeled_data_seperate, eval_data_seperate, all_client_num_in_total, client_num_in_total, labeled_idx, conver_point)

            logging.info(f"Infer. Gen{gen}: client_indexes is {client_indexes}, Labeled idx is {labeled_idx}")  

            for client in range(num_clients):
                for pattern_id in pattern_ids:
                    client_idx = client_indexes[client]
                    
                    if gen > conver_point and augmentation: 
                        train_data = np.array(train_data_sperate[client]).tolist()
                        unlabeled_data = np.array(unlabeled_data_seperate[client]).tolist()
                        eval_data = np.array(eval_data_seperate[client]).tolist()
                    else: # within conver_point and without augmentation, train_data_seperate will be fixed forever, and the client list is fixed, so it performs poor. We rearrange them as the client_idx mannually.
                        train_data = np.array(train_data_sperate[client_idx]).tolist()
                        unlabeled_data = np.array(unlabeled_data_seperate[client_idx]).tolist()
                        eval_data = np.array(eval_data_seperate[client_idx]).tolist()

                    logging.info(f"Client {client_idx}: len of train set: {len(train_data)}")

                    if gen > 0 and augmentation and pattern_id == pattern_ids[0] and gen % infer_freq == 0: # 是否利用unlabeled data, 只用第一个pattern训练出来的模型来增强，其他的用来验证
                        
                        # select the unlabeled data for inference by voting_k
                        # forked from https://github.com/HKUNLP/icl-selective-annotation

                        if augment == "curriculum":
                            augmented_point = min(int(len(unlabeled_data) * aug_data_point / 100 * gen/infer_freq), len(unlabeled_data))
                        else :
                            augmented_point = min(aug_data_point, len(unlabeled_data))
                        unlabeled_data_normal = [deepcopy(d) for d in unlabeled_data]
                        if random_filter:
                            select_num = int(len(unlabeled_data) * args.vote_k) if args.vote_k <= 1 else augmented_point
                            logging.info(f"Random filter the unlabeled data, select_num is {select_num}.")
                            np.random.seed(seed)
                            unlabeled_data = np.random.choice(unlabeled_data, select_num).tolist()
                            augmented_point = min(int(len(unlabeled_data) * aug_data_point / 100 * gen/infer_freq), len(unlabeled_data)) if args.vote_k <= 1 else augmented_point
                        elif args.vote_k > 0:
                            task_name = args.task_name
                            select_num = int(len(unlabeled_data) * args.vote_k) if args.vote_k <= 1 else augmented_point
                            logging.info(f"Client {client_idx}: len of unlabeled set: {len(unlabeled_data)}, select_num is {select_num}")

                            # sorted by the similarity&representive of unlabeled data themselves
                            if select == 'vote':
                                logging.info("Select via voting.")
                                unlabeled_data = select_by_voting(unlabeled_data,select_num, os.path.join(output_dir, f'g-1', f'client{client_idx}'), task_name)
                            elif select == 'sort':
                            # sorted by the similarity of labeled data
                                logging.info("Select via sorting.")
                                ipet_data_dir = os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data')
                                if not os.path.exists(ipet_data_dir):
                                    logging.info(f"Client {client_idx} has no ipet data.")
                                    ipet_data_dir = None
                                if ipet_data_dir:
                                    p = os.path.join(ipet_data_dir, 'train.bin')
                                    ipet_data = InputExample.load_examples(p)
                                    sample_num_list.append(original_data_size + ipet_data_size)
                                train_data_and_ipet_data = train_data + ipet_data if ipet_data_dir else train_data
                                unlabeled_data = select_by_sorting(train_data_and_ipet_data, unlabeled_data, select_num, task_name)

                            augmented_point = min(int(len(unlabeled_data) * aug_data_point / 100 * gen/infer_freq), len(unlabeled_data)) if args.vote_k <= 1 else augmented_point # if vote_k > 1, it means merely select those data to infer that could be selected for training.
                        else:
                            logging.info("Vote k filter is not activated.")
                        
                        models_path = []
                        aggregated_model_path_aug = os.path.join(output_dir, f'g{gen-1}',f'aggregated-p{pattern_id}')
                        
                        logging.info(f"Client {client_idx}: aggregated_model_path_aug is {aggregated_model_path_aug}")
                        wrapper = TransformerModelWrapper.from_pretrained(aggregated_model_path_aug)
                        
                        logging.info(f"Gen {gen}: Client {client_idx} will annotate {augmented_point} data.")
                        if len(unlabeled_data) > 0 and len(train_data) < augmented_point:
                            if args.vote_k > 0 and vote_k_check:
                                results = evaluate(wrapper, unlabeled_data_normal, ensemble_eval_config, label_list = ensemble_model_config.label_list)
                                logits = results['logits']
                                
                                logits_path = os.path.join(output_dir, f'g{gen-1}', f'client0-p{pattern_id}')
                                if not os.path.exists(logits_path):
                                    os.makedirs(logits_path)
                                save_logits(os.path.join(logits_path, 'logits.txt'), logits)
                                logging.info(f"Normal ipet selection results. Client {client_idx}: mean(max(logits)): {calculate_mean_max_logits(logits)}")

                                num_new_examples = augmented_point  - len(train_data)
                
                                generate_fedipet_train_sets(train_data=unlabeled_data_normal, unlabeled_data=unlabeled_data_normal,
                                                    labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                                    output_dir=os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data'), reduction=reduction,
                                                    num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                                    n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated, pattern=pattern_id)
                                p = os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data', 'train.bin')
                                ipet_train_data = InputExample.load_examples(p)
                                logging.info("Normal ipet selection results. (All unlabeled data is for inference)")
                                for ipet_train_example in ipet_train_data:
                                    uid = ipet_train_example.guid
                                    logging.info(f"Data {uid}'s logit is {ipet_train_example.logits}") 

                            results = evaluate(wrapper, unlabeled_data, ensemble_eval_config, label_list = ensemble_model_config.label_list)
                            logits = results['logits']
                            infer_sample_num_list.append(len(unlabeled_data))
                            
                            logits_path = os.path.join(output_dir, f'g{gen-1}', f'client0-p{pattern_id}')
                            if not os.path.exists(logits_path):
                                os.makedirs(logits_path)
                            save_logits(os.path.join(logits_path, 'logits.txt'), logits)

                            logging.info(f"Vote k selection results. Client {client_idx}: mean(max(logits)): {calculate_mean_max_logits(logits)}")
                            del wrapper
                            gc.collect()

                            num_new_examples = augmented_point  - len(train_data)

                            generate_fedipet_train_sets(train_data=unlabeled_data, unlabeled_data=unlabeled_data,
                                                labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                                output_dir=os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data'), reduction=reduction,
                                                num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                                n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated, pattern=pattern_id)
                            p = os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data', 'train.bin')
                            ipet_train_data = InputExample.load_examples(p)

                            if args.vote_k > 0 and vote_k_check:
                                logging.info("Vote k selection results. (Only part of unlabeled data is for inference)")
                                for ipet_train_example in ipet_train_data:
                                    uid = ipet_train_example.guid
                                    logging.info(f"Data {uid}'s logit is {ipet_train_example.logits}")

                            if client_idx not in labeled_idx and len(ipet_train_data) > 0:
                                # an int will be transferred to float after appending to a null list of numpy
                                labeled_idx = labeled_idx.tolist()
                                labeled_idx.append(client_idx)
                                labeled_idx = np.array(labeled_idx, dtype=int)

        if staleness or not augmentation:
            train_data_sperate, unlabeled_data_seperate, eval_data_seperate, curr_sample_num_list, client_indexes, num_clients = train_client_selection(gen, augmentation, train_data_all, unlabeled_data_all, eval_data_all, train_data_sperate, unlabeled_data_seperate, eval_data_seperate, all_client_num_in_total, client_num_in_total, labeled_idx, conver_point)
        

        logging.info(f"Train. Gen{gen}: client_indexes is {client_indexes}, Labeled idx is {labeled_idx}")   
            
        # Prompt local training
        sample_num_list = []
        for client in range(num_clients):
            for pattern_id in pattern_ids:
                client_idx = client_indexes[client]
                gen_output_dir_pattern = os.path.join(output_dir, f'g{gen}', f'client{client}-p{pattern_id}')
                
                train_data = np.array(train_data_sperate[client]).tolist()
                unlabeled_data = np.array(unlabeled_data_seperate[client]).tolist()
                eval_data = np.array(eval_data_seperate[client]).tolist()

                logging.info(f"Client {client_idx}: len of train set: {len(train_data)}")
                     
                # Step 2: Train an ensemble of models corresponding to individual clients (pattern = 1)
                
                ipet_data_dir = os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data')
                if not os.path.exists(ipet_data_dir):
                    logging.info(f"Client {client_idx} has no ipet data.")
                    ipet_data_dir = None

                if pattern_id == pattern_ids[0]: # only count for once
                    original_data_size = len(train_data)
                    ipet_data_size = 0
                    if ipet_data_dir:
                        p = os.path.join(ipet_data_dir, 'train.bin')
                        ipet_data_size = len(InputExample.load_examples(p))
                    sample_num_list.append(original_data_size + ipet_data_size)
                
                aggregated_model_path_pattern = None
                if gen > 0 and len(sample_num_list) > 0:
                    aggregated_model_path_pattern = f"{aggregated_model_path}-p{pattern_id}"

                train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_id,
                                gen_output_dir_pattern, ipet_data_dir=ipet_data_dir,
                                repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                                eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=augmentation, aggregated_model_path = aggregated_model_path_pattern, check_data=check_data, limit=limit) 
        
        logging.info("The current generation of Gen{}: sample_num_list is {}".format(gen, sample_num_list))

        # Model aggregation
        aggregated_model_path = os.path.join(output_dir, f'g{gen}',f'aggregated')
        if len(sample_num_list) > 0: # Compatible for zero-shot learning
            for pattern_id in pattern_ids:
                models_path = []
                aggregated_model_path_pattern = f"{aggregated_model_path}-p{pattern_id}"
                for i in range(num_clients):
                    if i == len(sample_num_list):
                        break
                    
                    pattern_iter_input_dir = os.path.join(output_dir, f'g{gen}',f'client{i}-p{pattern_id}')
                    models_path.append(pattern_iter_input_dir)
                    if debug:
                        logging.info(f"{pattern_iter_input_dir} is going to be aggregated with weight count {sample_num_list[i]}.")

                fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                wrapper = init_model(ensemble_model_config)
                wrapper.model = fl_model
                if debug:
                    logging.info("Saving aggregated trained model at {}".format(aggregated_model_path_pattern))
                wrapper.save(aggregated_model_path_pattern) 
        else:
            for pattern_id in pattern_ids:
                models_path = []
                aggregated_model_path_pattern = f"{aggregated_model_path}-p{pattern_id}"
                wrapper = init_model(ensemble_model_config)
                logging.info("Zero-shot: saving origin pretrained model at {}".format(aggregated_model_path_pattern))
                wrapper.save(aggregated_model_path_pattern) 

        logging.info("------INFO------")
        logging.info(f'Gen {gen}:  labeled_idx = {labeled_idx}, len(labeled_idx) = {len(labeled_idx)}')
        logging.info(f"Gen {gen}:  train data = {sample_num_list}")
        logging.info(f"Gen {gen}:  infer data = {infer_sample_num_list}")

        if do_eval:
            if eval_step > 1: # Only eval when gen = 0, 10, 20... per 10 gens
                if gen % eval_step == 0:
                    eval_result = []
                    if merge_eval:
                        eval_data_all_merged = np.concatenate(np.array(eval_data_all))
                        eval_result.append(evaluate(wrapper, eval_data_all_merged, ensemble_eval_config, label_list = ensemble_model_config.label_list)['scores']['acc'])
                    else:
                        for i in range(all_client_num_in_total): # eval aggregated performance on all eval set
                            eval_result.append(evaluate(wrapper, eval_data_all[i], ensemble_eval_config, label_list = ensemble_model_config.label_list)['scores']['acc'])
                            logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))
                    

                    if debug:
                        logging.info("All clients' eval performance results is:")
                        logging.info(eval_result)

                    logging.info('Acc iter is {}. Gen {} aggregated model performance is: {}'.format(gen, gen // eval_step, np.mean(np.array(eval_result))))
            
            else:# Normal
                eval_result = []
                if merge_eval:
                    eval_data_all_merged = np.concatenate(np.array(eval_data_all))
                    eval_result.append(evaluate(wrapper, eval_data_all_merged, ensemble_eval_config, label_list = ensemble_model_config.label_list)['scores']['acc'])
                    # logging.info("Gen {}: Client {} eval acc is: {}".format(gen-1, i, eval_result[-1]))
                else:
                    for i in range(all_client_num_in_total): # eval aggregated performance on all eval set
                        eval_result.append(evaluate(wrapper, eval_data_all[i], ensemble_eval_config, label_list = ensemble_model_config.label_list)['scores']['acc'])
                        logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))

                if debug:
                    logging.info("All clients' eval performance results is:")
                    logging.info(eval_result)

                logging.info('Gen {} aggregated model performance is: {}'.format(gen, np.mean(np.array(eval_result))))
            
            del wrapper
            gc.collect()

        

def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_ids: List[int], output_dir: str, ipet_data_dir: str = None, repetitions: int = 3,
                       train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       save_unlabeled_logits: bool = False, seed: int = 42, aggregated_model_path: str = None, last_iteration_model_path: str = None, check_data: List[InputExample] = None, limit: int = 0):
    """
    Train and evaluate an ensemble of PET models without knowledge distillation.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param pattern_ids: the ids of all PVPs to use
    :param output_dir: the output directory
    :param ipet_data_dir: optional directory containing additional training data for iPET
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param save_unlabeled_logits: whether logits for unlabeled examples should be saved in a file ``logits.txt``. This
           is required for both iPET and knowledge distillation.
    :param seed: the random seed to use
    """

    results = defaultdict(lambda: defaultdict(list))
    set_seed(seed)

    for iteration in range(repetitions):

        model_config.pattern_id = pattern_ids
        results_dict = {}

        pattern_iter_output_dir = f"{output_dir}"

        if not os.path.exists(pattern_iter_output_dir):
            os.makedirs(pattern_iter_output_dir)

        wrapper = None

        # Training
        if do_train:

            wrapper = init_model(model_config)
            
            # Using the previous aggregated model
            if aggregated_model_path:
                logging.info("Loading previous aggregated trained model from {}".format(aggregated_model_path))
                wrapper = TransformerModelWrapper.from_pretrained(aggregated_model_path)
            
            # Using the previous private model for each client seperatedly.
            if last_iteration_model_path:
                logging.info("Loading previous private trained model in last iteration for each client seperatedly from {}.".format(last_iteration_model_path))
                wrapper = TransformerModelWrapper.from_pretrained(last_iteration_model_path)

            if ipet_data_dir:
                p = os.path.join(ipet_data_dir, 'train.bin')
                if os.path.exists(p):
                    ipet_train_data = InputExample.load_examples(p)
                else:
                    ipet_train_data = None
                logging.info(f"Evaluating soft label on {ipet_data_dir}")

                # vanilla annotating
                if check_eval:
                    ipet_train_data = eval_softlabel(ipet_train_data, check_data, replace=correct_label, limit=limit)
                else:
                    logging.info("Not using check data for annotating.")
            else:
                ipet_train_data = None

            results_dict.update(train_single_model(wrapper, train_data, train_config, eval_config,
                                                    ipet_train_data=ipet_train_data,
                                                    unlabeled_data=unlabeled_data, eval_data = eval_data))

            with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                fh.write(str(results_dict))

            logging.info("Saving trained model at {}".format(pattern_iter_output_dir))
            wrapper.save(pattern_iter_output_dir)
            train_config.save(os.path.join(pattern_iter_output_dir, 'train_config.json'))
            eval_config.save(os.path.join(pattern_iter_output_dir, 'eval_config.json'))
            logging.info("Saving complete")

            # if save_unlabeled_logits:
            #     logits = evaluate(wrapper, unlabeled_data, eval_config)['logits']
            #     save_logits(os.path.join(pattern_iter_output_dir, 'logits.txt'), logits)

            if not do_eval:
                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

        # Evaluation
        if do_eval:
            logging.info("Starting evaluation...")
            if not wrapper:
                logging.info("Loading from pattern_iter_output_dir")
                wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

            eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)

            save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
            save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

            scores = eval_result['scores']

            if debug:
                logging.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_ids, iteration))
                logging.info(f"Eval results on this client's local data: f{scores}")
                # logging.info(pattern_iter_output_dir)

            results_dict['test_set_after_training'] = scores
            with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                json.dump(results_dict, fh)

            # for metric, value in scores.items():
            #     results[metric][pattern_ids].append(value)

            wrapper.model = None
            wrapper = None
            torch.cuda.empty_cache()

    # if do_eval:
    #     logging.info("=== OVERALL RESULTS ===")
    #     _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    # else:
    #     logging.info("=== ENSEMBLE TRAINING COMPLETE ===")


def train_single_model(model: TransformerModelWrapper, train_data: List[InputExample], config: TrainConfig,
                       eval_config: EvalConfig = None, ipet_train_data: List[InputExample] = None,
                       unlabeled_data: List[InputExample] = None, return_train_set_results: bool = True, eval_data: List[InputExample] = None):
    """
    Train a single model.

    :param model: the model to train
    :param train_data: the training examples to use
    :param config: the training config
    :param eval_config: the evaluation config
    :param ipet_train_data: an optional list of iPET training examples to use
    :param unlabeled_data: an optional list of unlabeled examples to use
    :param return_train_set_results: whether results on the train set before and after training should be computed and
           returned
    :return: a dictionary containing the global step, average loss and (optionally) results on the train set
    """

    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")
    if not ipet_train_data:
        ipet_train_data = []

    results_dict = {}

    model.model.to(device)
    
    if debug:
        # debug mode, evaluate the val set, default is train_data
        results_dict['train_set_before_training'] = evaluate(model, eval_data, eval_config)['scores']['acc']

        
        logging.info('init acc: val acc before training is {}'.format(results_dict['train_set_before_training']))

        logging.info("--------------------------------------------")
        logging.info("train_data is:")
        logging.info(train_data)

        logging.info("--------------------------------------------")
        logging.info("ipet_train_data is:")
        logging.info(ipet_train_data)

    all_train_data = train_data + ipet_train_data

    logging.info('len of all_train_data: {}'.format(len(all_train_data)))

    if not all_train_data and not config.use_logits:
        logging.warning('Training method was called without training examples')
    else:
        if bitfit_training:
            model = deactivate_relevant_gradients(model)
        global_step, tr_loss = model.train(
            all_train_data, device,
            per_gpu_train_batch_size=config.per_gpu_train_batch_size,
            per_gpu_unlabeled_batch_size=config.per_gpu_unlabeled_batch_size,
            n_gpu=config.n_gpu,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            adam_epsilon=config.adam_epsilon,
            warmup_steps=config.warmup_steps,
            max_grad_norm=config.max_grad_norm,
            unlabeled_data=unlabeled_data if config.lm_training or config.use_logits else None,
            lm_training=config.lm_training,
            use_logits=config.use_logits,
            alpha=config.alpha,
            temperature=config.temperature,
            task_eval_data=eval_data
        )
        results_dict['global_step'] = global_step
        results_dict['average_loss'] = tr_loss

    # if train_data and return_train_set_results:
    #     results_dict['train_set_after_training'] = evaluate(model, eval_data, eval_config)['scores']['acc']

    return results_dict


def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             priming_data: List[InputExample] = None, label_list: List[InputExample] = None) -> Dict:
    """
    Evaluate a model.

    :param model: the model to evaluate
    :param eval_data: the examples for evaluation
    :param config: the evaluation config
    :param priming_data: an optional list of priming data to use
    :return: a dictionary containing the model's logits, predictions and (if any metrics are given) scores
    """

    if config.priming:
        for example in eval_data:
            example.meta['priming_data'] = priming_data
    
    metrics = config.metrics if config.metrics else ['acc']
    device = torch.device(config.device if config.device else "cuda" if torch.cuda.is_available() else "cpu")

    model.model.to(device)
    results = model.eval(eval_data, device, per_gpu_eval_batch_size=config.per_gpu_eval_batch_size,
                         n_gpu=config.n_gpu, decoding_strategy=config.decoding_strategy, priming=config.priming)

    predictions = np.argmax(results['logits'], axis=1)
    scores = {}

    if label_list and debug:
        get_prediction_accuracy_distribution(predictions, results['labels'], label_list)

    for metric in metrics:
        if metric == 'acc':
            scores[metric] = simple_accuracy(predictions, results['labels'])
        elif metric == 'f1':
            scores[metric] = f1_score(results['labels'], predictions)
        elif metric == 'f1-macro':
            scores[metric] = f1_score(results['labels'], predictions, average='macro')
        elif metric == 'em':
            scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
        else:
            raise ValueError(f"Metric '{metric}' not implemented")

    results['scores'] = scores
    results['predictions'] = predictions
    return results


def merge_logits(logits_dir: str, output_file: str, reduction: str):
    """
    Merge the logits predicted for unlabeled examples by multiple models.

    :param logits_dir: a directory for which each sub-directory corresponds to a pretrained model and contains
           both a file ``results.txt`` containing that model's results on the training set and a file ``logits.txt``
           containing that model's predictions for the unlabeled data.
    :param output_file: the file to which the merged logits for all unlabeled examples are written.
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    """
    subdirs = next(os.walk(logits_dir))[1]
    logging.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    all_logits_lists = []

    for subdir in subdirs:
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(results_file) or not os.path.exists(logits_file):
            logging.warning(f"Skipping subdir '{subdir}' because 'results.txt' or 'logits.txt' not found")
            continue

        if reduction == 'mean':
            result_train = 1
        else:
            with open(results_file, 'r') as fh:
                results = ast.literal_eval(fh.read())
                result_train = results['train_set_before_training']

        with open(logits_file, 'r') as fh:
            for line in fh.read().splitlines():
                example_logits = [float(x) for x in line.split()]
                logits.append(example_logits)

        logging.info("File {}: Score = {}, #Logits = {}, #Labels = {}".format(
            results_file, result_train, len(logits), len(logits[0])))

        loglist = LogitsList(score=result_train, logits=logits)
        all_logits_lists.append(loglist)

    merged_loglist = merge_logits_lists(all_logits_lists, reduction=reduction)
    merged_loglist.save(output_file)



def train_fedclassifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42, beta: int = None, client_num_in_total: int = None, check_data: List[InputExample] = None, all_client_num_in_total: int = None, labeled_idx: List[int] = None, augmentation: bool = True, aug_data_point: int = 100):
    """
    Train and evaluate a sequence classification model.

    :param model_config: the model configuration to use
    :param train_config: the training configuration to use
    :param eval_config: the evaluation configuration to use
    :param output_dir: the output directory
    :param repetitions: the number of training repetitions
    :param train_data: the training examples to use
    :param unlabeled_data: the unlabeled examples to use
    :param eval_data: the evaluation examples to use
    :param do_train: whether to perform training
    :param do_eval: whether to perform evaluation
    :param seed: the random seed to use
    """

    train_config.use_logits=False # !!!!!!!!!!!! to be the same with sequence classifer

    num_clients = min(5, client_num_in_total) 
    
    train_data_sperate, unlabeled_data_seperate, eval_data_seperate = find_labeled(labeled_idx, train_data, unlabeled_data, eval_data)

    eval_data_all = eval_data

    aggregated_model_path = None

    for gen in range(repetitions):
        delete_cache(gen, output_dir)
        # Select clients
        np.random.seed(gen)
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("Gen {}: client_indexes is {}".format(gen, client_indexes))
        
        # Calculate the samples numbers of each clients involved in this round 
        sample_num_list = np.array([])
        for client in range(num_clients):
            sample_num_list = np.append(sample_num_list, len(train_data_sperate[client_indexes[client]]))
        logging.info("Gen{}: sample_num_list is {}".format(gen, sample_num_list))

       
        infer_sample_num_list = []*len(sample_num_list)
        for client in range(num_clients):
            client_idx = client_indexes[client]
            gen_output_dir = os.path.join(output_dir, f'g{gen}', f'client{client}')
            train_data = np.array(train_data_sperate[client_idx]).tolist()
            unlabeled_data = np.array(unlabeled_data_seperate[client_idx]).tolist()
            eval_data = np.array(eval_data_seperate[client_idx]).tolist()

            
            if gen > 0 and augmentation:
                aggregated_model_path_aug = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
                wrapper = TransformerModelWrapper.from_pretrained(aggregated_model_path_aug)
                augmented_point = min(aug_data_point, len(unlabeled_data))
                if len(unlabeled_data) > 0 and len(train_data) < augmented_point:
                    results = evaluate(wrapper, unlabeled_data, eval_config, label_list = model_config.label_list)
                    logits = results['logits']
                    infer_sample_num_list.append(len(unlabeled_data))
                    
                    logits_path = os.path.join(output_dir, f'g{gen-1}', f'client0-p0')
                    if not os.path.exists(logits_path):
                        os.makedirs(logits_path)
                    save_logits(os.path.join(logits_path, 'logits.txt'), logits)

                    logging.info(f"Vote k selection results. Client {client_idx}: mean(max(logits)): {calculate_mean_max_logits(logits)}")
                    del wrapper
                    gc.collect()

                    num_new_examples = augmented_point  - len(train_data)

                    generate_fedipet_train_sets(train_data=unlabeled_data, unlabeled_data=unlabeled_data,
                                        labels=model_config.label_list, logits_percentage = 100, reduction= 'mean',logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                        output_dir=os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data'),
                                        num_new_examples=num_new_examples,
                                        n_most_likely=-1, seed=seed,pattern=0)
                
            

            
            ipet_data_dir = os.path.join(output_dir, f'g-1', f'client{client_idx}', 'this-gen-train-data') if gen > 0 and augmentation else None
            train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[1], output_dir=gen_output_dir, ipet_data_dir=ipet_data_dir,
                            repetitions=1,
                            train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data, do_train=do_train,
                            do_eval=do_eval, seed=seed, aggregated_model_path=aggregated_model_path, check_data=check_data)

        aggregated_model_path = os.path.join(output_dir, f'g{gen}',f'aggregated')
        # Aggergate models trained in current round.
        if len(sample_num_list) > 0: # compatible for zero-shot learning
            models_path = []
            for i in range(num_clients):
                pattern_iter_input_dir = os.path.join(output_dir, f'g{gen}',f'client{i}')
                models_path.append(pattern_iter_input_dir)
                if not os.path.exists(pattern_iter_input_dir):
                    os.makedirs(pattern_iter_input_dir)
            fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
            wrapper = init_model(model_config)
            wrapper.model = fl_model

            logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
            wrapper.save(aggregated_model_path)
        else:
            wrapper = init_model(model_config)
            logging.info("Zero-shot: saving origin pretrained model at {}".format(aggregated_model_path))
            wrapper.save(aggregated_model_path) 
        
        logging.info("------INFO------")
        logging.info(f'Gen {gen}:  labeled_idx = {labeled_idx}, len(labeled_idx) = {len(labeled_idx)}')
        logging.info(f"Gen {gen}:  train data = {sample_num_list}")
        logging.info(f"Gen {gen}:  infer data = {infer_sample_num_list}")
            
        if eval_step > 1:# Only eval when gen = 0, 10, 20... per 10 gens
            if gen % 10 == 0:
                eval_result = []
                if merge_eval:
                    eval_data_all_merged = np.concatenate(np.array(eval_data_all))
                    eval_result.append(evaluate(wrapper, eval_data_all_merged, eval_config, label_list = model_config.label_list)['scores']['acc'])
                    logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))
                else:
                    for i in range(all_client_num_in_total): # eval aggregated performance on all eval set
                        eval_result.append(evaluate(wrapper, eval_data_all[i], eval_config, label_list = model_config.label_list)['scores']['acc'])
                        logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))

                if debug:
                    logging.info("All clients' eval performance results is:")
                    logging.info(eval_result)

                logging.info('Acc iter is {}. Gen {} aggregated model performance is: {}'.format(gen, gen // eval_step, np.mean(np.array(eval_result))))

        else: # Normal
            eval_result = []
            for i in range(all_client_num_in_total): # eval aggregated performance on all eval set
                eval_result.append(evaluate(wrapper, eval_data_all[i], eval_config, label_list = model_config.label_list)['scores']['acc'])
                # logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))

            if debug:
                logging.info("All clients' eval performance results is:")
                logging.info(eval_result)

            logging.info('Gen {} aggregated model performance is: {}'.format(gen, np.mean(np.array(eval_result))))
        
        del wrapper
        gc.collect()