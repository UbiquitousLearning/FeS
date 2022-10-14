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
import ast
from enum import Flag
import json
import os
import random
import statistics
from abc import ABC
from collections import defaultdict
from copy import deepcopy
from this import s
from typing import List, Dict

import numpy as np
from sklearn import ensemble
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import logging

from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

import transformers
transformers.logging.set_verbosity_error()

import gc

from fed.model import *
from fed.augment import *
from fed.utils import *

process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

debug = False
eval_step = 10
merge_eval = False
correct_label = False
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
               augmentation: bool = True, fed: bool = True, vanilla: bool = True, beta: int = None, client_num_in_total: int = None, check_data: List[InputExample] = None, all_client_num_in_total: int = None, labeled_idx: List[int] = None, aug_data_point: int = 100, conver_point: int = 0, limit: int = 0):
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
        
        # get the sample num list of the last iteration for fedavg aggregation
        # those train_data_seperate will be different in the second round, and thus leading to null error
        if gen > 0:
            # sample_num_list was inherited from the last iteration

            logging.info("The prior generation of Gen{}: sample_num_list is {}".format(gen, sample_num_list))

            # client selection
            train_data_sperate, unlabeled_data_seperate, eval_data_seperate, curr_sample_num_list, client_indexes, num_clients = client_selection(gen, augmentation, train_data_all, unlabeled_data_all, eval_data_all, train_data_sperate, unlabeled_data_seperate, eval_data_seperate, all_client_num_in_total, client_num_in_total, labeled_idx, conver_point)

        
        else: 
            train_data_sperate, unlabeled_data_seperate, eval_data_seperate, sample_num_list, client_indexes, num_clients = client_selection(gen, augmentation, train_data_all, unlabeled_data_all, eval_data_all, train_data_sperate, unlabeled_data_seperate, eval_data_seperate, all_client_num_in_total, client_num_in_total, labeled_idx, conver_point)

            logging.info("Gen{}: sample_num_list is {}".format(gen, sample_num_list))        
        
        # Data augmentation
        sample_num_list = []
        for client in range(num_clients):
            for pattern_id in pattern_ids:
                client_idx = client_indexes[client]
                gen_output_dir_pattern = os.path.join(output_dir, f'g{gen}', f'client{client}-p{pattern_id}')
                
                if gen > conver_point and augmentation: 
                    train_data = np.array(train_data_sperate[client]).tolist()
                    unlabeled_data = np.array(unlabeled_data_seperate[client]).tolist()
                    eval_data = np.array(eval_data_seperate[client]).tolist()
                else: # within conver_point and without augmentation, train_data_seperate will be fixed forever, and the client list is fixed, so it performs poor
                    train_data = np.array(train_data_sperate[client_idx]).tolist()
                    unlabeled_data = np.array(unlabeled_data_seperate[client_idx]).tolist()
                    eval_data = np.array(eval_data_seperate[client_idx]).tolist()

                logging.info(f"Client {client_idx}: len of train set: {len(train_data)}")

                if gen > 0 and augmentation and pattern_id == pattern_ids[0]: # 是否利用unlabeled data, 只用第一个pattern训练出来的模型来增强，其他的用来验证
                    models_path = []
                    aggregated_model_path_aug = os.path.join(output_dir, f'g{gen-1}',f'aggregated-p{pattern_id}')
                    
                    wrapper = TransformerModelWrapper.from_pretrained(aggregated_model_path_aug)

                    if len(unlabeled_data) > 0 and len(train_data) < aug_data_point:
                        results = evaluate(wrapper, unlabeled_data, ensemble_eval_config, label_list = ensemble_model_config.label_list)
                        logits = results['logits']
                        
                        logits_path = os.path.join(output_dir, f'g{gen-1}', f'client0-p{pattern_id}')
                        if not os.path.exists(logits_path):
                            os.makedirs(logits_path)
                        save_logits(os.path.join(logits_path, 'logits.txt'), logits)

                        logging.info(f"Client {client_idx} save_logits done")
                        del wrapper
                        gc.collect()

                        num_new_examples = aug_data_point - len(train_data)
                        # num_new_examples = min(aug_data_point, int(2 ** (gen + 1) - len(train_data))) # 由原先的**级别增长，降至*级别增长
                        generate_fedipet_train_sets(train_data=unlabeled_data, unlabeled_data=unlabeled_data,
                                            labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                            output_dir=os.path.join(output_dir, f'g{gen}', f'client{client_idx}', 'this-gen-train-data'), reduction=reduction,
                                            num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                            n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated, pattern=pattern_id)
                                            
                # Step 2: Train an ensemble of models corresponding to individual clients (pattern = 1)
                ipet_data_dir = os.path.join(output_dir, f'g{gen}', f'client{client_idx}', 'this-gen-train-data') if gen > 0 and augmentation and len(unlabeled_data) > 0 and len(train_data) < aug_data_point else None

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
                    logging.info(f"{pattern_iter_input_dir} is going to be aggregated with weight count {sample_num_list[i]}.")

                fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                wrapper = init_model(ensemble_model_config)
                wrapper.model = fl_model
                logging.info("Saving aggregated trained model at {}".format(aggregated_model_path_pattern))
                wrapper.save(aggregated_model_path_pattern) 
        else:
            for pattern_id in pattern_ids:
                models_path = []
                aggregated_model_path_pattern = f"{aggregated_model_path}-p{pattern_id}"
                wrapper = init_model(ensemble_model_config)
                logging.info("Zero-shot: saving origin pretrained model at {}".format(aggregated_model_path_pattern))
                wrapper.save(aggregated_model_path_pattern) 

        # do_eval = False
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
                ipet_train_data = InputExample.load_examples(p)
                # for example in ipet_train_data:
                #     example.logits = None
                
                # ipet_train_data = [deepcopy(ex) for ex in check_data[:996]]
                
                logging.info("Evaluating soft label~")

                ipet_train_data = eval_softlabel(ipet_train_data, check_data, replace=correct_label, limit=limit)

                # ensemble voting
                # pattern_ids = [0]
                # labels_pattern = []
                # for i in range(len(pattern_ids)):
                #     pattern_id = pattern_ids[i]
                #     aggregated_model_path_pattern = aggregated_model_path.split('-')[0] + "-" + aggregated_model_path.split('-')[1] + f'-p{pattern_id}'
                #     wrapper = TransformerModelWrapper.from_pretrained(aggregated_model_path_pattern)
                #     results = evaluate(wrapper, ipet_train_data, eval_config, model_config.label_list)
                #     # logging.info(results)
                #     label = []
                #     for l in results['logits']:
                #         label.append(model_config.label_list[np.argmax(l).item()])
                #     labels_pattern.append(label)
                #     # logging.info(labels_pattern)

                # ipet_train_data = eval_softlabel(ipet_train_data, check_data, replace=correct_label, labels = labels_pattern)
            else:
                ipet_train_data = None
            
            # # See test acc before training
            # eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)

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
            logging.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_ids, iteration))
            logging.info(f"Eval results on this client's local data: f{scores}")
            logging.info(pattern_iter_output_dir)

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


def merge_logits_lists(logits_lists: List[LogitsList], reduction: str = 'mean') -> LogitsList:
    """
    Merge a list of :class:`LogitsList` objects.

    :param logits_lists: the lists to merge
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :return: the merged list
    """

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    return LogitsList(score=-1, logits=logits)

                                   
def generate_fedipet_train_sets(train_data: List[InputExample], unlabeled_data: List[InputExample], labels: List[str],
                             logits_dir: str, output_dir: str, reduction: str, num_new_examples: int,
                             logits_percentage: float, n_most_likely: int = -1, seed: int = 42, aggregated: bool = True, pattern: int = None):
    """
    Generate training sets for the next generation of iPET models.

    :param train_data: the training examples
    :param unlabeled_data: the unlabeled examples
    :param labels: the list of all possible labels
    :param logits_dir: the directory that contains the predictions of all models in the current generation for the
           unlabeled data.
    :param output_dir: the output directory
    :param reduction: the strategy for merging logits, either 'mean' or 'wmean'. For 'mean', all models contribute
           equally, for 'wmean', each model's contribution is proportional to its accuracy on the training set before
           training.
    :param num_new_examples: the number of new examples to create
    :param logits_percentage: the percentage of models to use for annotating training sets for the next generation
    :param n_most_likely: If >0, in the first generation the n_most_likely examples per label are chosen even
                              if their predicted label is different
    :param seed: the random seed to use
    """
    subdirs = next(os.walk(logits_dir))[1]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Found the following {} subdirectories: {}".format(len(subdirs), subdirs))

    if train_data:
        train_examples_per_label = [sum(1 for ex in train_data if ex.label == label) for label in labels]
        multiplier = num_new_examples / len(train_data)
        examples_per_label = [int(epl * multiplier) for epl in train_examples_per_label]
        logging.info(f"Example distribution in the original dataset: {train_examples_per_label}")
    else:
        examples_per_label = eq_div(num_new_examples, len(labels))

    logging.info(f"Target distribution for the new dataset: {examples_per_label}")

    for example in unlabeled_data:
        example.label, example.logits = None, None

    logits_lists = {}

    rng = random.Random(seed)
    rng_np = np.random.RandomState(seed)
    
    for subdir in subdirs:
        if pattern:
            s = f'client0-p{pattern}'
        else:
            s = 'client0-p0'
        if aggregated:
            if subdir != s:
                # print(subdir,' ','no aggregated data')
                continue
        # print(subdir,' ','has aggregated data')
        results_file = os.path.join(logits_dir, subdir, 'results.txt')
        logits_file = os.path.join(logits_dir, subdir, 'logits.txt')
        logits = []

        if not os.path.exists(logits_file) :
            logging.warning(f"Skipping subdir '{subdir}' because 'logits.txt' not found")
            continue

        if not os.path.exists(results_file) :
            logging.warning(f"Setting reduction as mean because 'results.txt' not found")
            result_train = 1


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
        logits_lists[subdir] = loglist


    other_logits_lists = [ll for sd, ll in logits_lists.items()]
    subdir_train_set = generate_ipet_train_set(
        other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
        logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
        rng_np=rng_np
    )

    InputExample.save_examples(subdir_train_set,
                                os.path.join(output_dir, 'train.bin'))
    logging.info(f'Save data with soft label to {output_dir}')

def generate_ipet_train_set(logits_lists: List[LogitsList], labels: List[str], original_data: List[InputExample],
                            examples_per_label: List[int], logits_percentage: float, reduction: str = 'mean',
                            n_most_likely: int = -1, rng=None, rng_np=None, aggregated: bool = True) -> List[InputExample]:
    """
    Generate a single training set for the next generation of iPET models.

    :param logits_lists: predictions from the previous generation of models
    :param labels: all task labels
    :param original_data: the original training data corresponding to the logits_lists
    :param examples_per_label: the number of examples per label to create
    :param logits_percentage: the percentage of models/logits to choose
    :param reduction: the reduction strategy ('wmean' or 'mean')
    :param n_most_likely: if >0, for each label the n_most_likely examples with the highest logits are chosen
    :param rng: the random number generator to use for non-numpy operations
    :param rng_np: the random number generator to use for numpy operations
    :return: a list of input examples that serves as training set for the next generation
    """
    # logging.info(logits_lists)
    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

    n_most_likely = 100

    if not rng:
        rng = random.Random()
    if not rng_np:
        rng_np = np.random.RandomState()

    if aggregated:
        num_logits_lists = 1
    else:
        num_logits_lists = round(len(logits_lists) * logits_percentage) 
    logits_lists = rng.sample(logits_lists, k=num_logits_lists)
    logits = np.array([ll.logits for ll in logits_lists])
    weights = np.array([ll.score for ll in logits_lists])

    if reduction == 'mean':
        logits = np.mean(logits, axis=0)
        logits = softmax(logits, axis=1).tolist()
    elif reduction == 'wmean':
        logits = np.average(logits, axis=0, weights=weights)
        logits = softmax(logits, axis=1).tolist()
    else:
        raise ValueError("Reduction strategy '{}' not implemented".format(reduction))

    assert len(logits) == len(original_data)

    for lgs, example in zip(logits, original_data):
        example.logits = lgs
        example.label = labels[np.argmax(example.logits).item()]

    test_set = []

    for idx, label in enumerate(labels):

        if n_most_likely <= 0:
            examples = [ex for ex in original_data if ex.label == label]
            logging.info("There are {} examples for label {}".format(len(examples), label))
            while len(examples) < examples_per_label[idx]:
                # upsample examples if there are too few
                examples.extend(ex for ex in original_data if ex.label == label)
        else:
            examples = [(ex.logits[idx], ex_idx, ex) for ex_idx, ex in enumerate(original_data)]
            examples.sort(reverse=True)
            examples = [ex for score, ex_idx, ex in examples[:n_most_likely]]
            examples = [deepcopy(ex) for ex in examples]
            for example in examples:
                example.logits = [example.logits[idx]]
                example.label = label

        label_examples = _draw_examples_by_label_probability(
            examples=examples, num_examples=examples_per_label[idx], rng=rng_np)
        test_set.extend(label_examples)

    return test_set


def _draw_examples_by_label_probability(examples: List[InputExample], num_examples: int, rng) -> List[InputExample]:
    if len(examples) > 0:
        label_probabilities = [max(example.logits) for example in examples]
        # logging.info(f'Label_probabilities NAN is {np.isnan(label_probabilities)}')
        # label_probabilities[np.isnan(label_probabilities)] = np.mean(label_probabilities)
        sum_label_probabilities = sum(label_probabilities)
        label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
        return rng.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()
        # return rng.choice(examples, size=num_examples, replace=False).tolist() # do not choose those high probabilities.
    else:
        logging.info("This client has 0 examples to draw. Funtion _draw_examples_by_label_probability is passed.")
        return []


def train_fedclassifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42, beta: int = None, client_num_in_total: int = None, check_data: List[InputExample] = None, all_client_num_in_total: int = None, labeled_idx: List[int] = None):
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

        for client in range(num_clients):
            client_idx = client_indexes[client]
            gen_output_dir = os.path.join(output_dir, f'g{gen}', f'client{client}')
            train_data = np.array(train_data_sperate[client_idx]).tolist()
            unlabeled_data = np.array(unlabeled_data_seperate[client_idx]).tolist()
            eval_data = np.array(eval_data_seperate[client_idx]).tolist()

            train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[1], output_dir=gen_output_dir,
                            repetitions=1,
                            train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data, do_train=do_train,
                            do_eval=do_eval, seed=seed, aggregated_model_path=aggregated_model_path, check_data=check_data)

        aggregated_model_path = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
        # Aggergate models trained in current round.
        if len(sample_num_list) > 0: # compatible for zero-shot learning
            models_path = []
            for i in range(num_clients):
                pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')
                models_path.append(pattern_iter_input_dir)
            fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
            wrapper = init_model(model_config)
            wrapper.model = fl_model

            logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
            wrapper.save(aggregated_model_path)
        else:
            wrapper = init_model(model_config)
            logging.info("Zero-shot: saving origin pretrained model at {}".format(aggregated_model_path))
            wrapper.save(aggregated_model_path) 
            
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
                logging.info("Gen {}: Client {} eval acc is: {}".format(gen, i, eval_result[-1]))

            if debug:
                logging.info("All clients' eval performance results is:")
                logging.info(eval_result)

            logging.info('Gen {} aggregated model performance is: {}'.format(gen-1, np.mean(np.array(eval_result))))
        
        del wrapper
        gc.collect()