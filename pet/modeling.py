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
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers.data.metrics import simple_accuracy

import logging
from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
from pet.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

import collections

import transformers
transformers.logging.set_verbosity_error()

process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

debug = True
# vanilla = False # whether fed vanilla is on, fed vanilla means no augmentation, but is fed, means using ft instead of pl to train local model, and aggregate the model via fedavg
# aggregated = True # 是否将10个client训练出来的模型fedavg一下，只infer一次;后面的两个函数里也要改一下
# augmentation = False # 如果不开augmentation，那每个client只能依靠自己的数据来进行训练，而且也不会用到unlabeled data (origin), fed is off
# fed = False

def partition_class_samples_with_dirichlet_distribution(
        train_data, beta, client_num,seed):
    """
    params
    ------------------------------------
    train_data : labeled train dataset
    beta : int  similarity of each client, the larger the beta the similar data for each client
    client_num : int number of clients
    seed : random seed, from initial args; default is 42
    ------------------------------------

    return
    ------------------------------------
    train_data_dirichlet : labeled train dataset with dirichlet distribution (not uniformed)
    ------------------------------------
    """
    np.random.seed(seed)
    # train dataset should be shuffled before
    N = len(train_data)

    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(beta, client_num))

    train_data_dirichlet_list = np.array([])

    N_available = N - client_num # ensure that each client has 1 samples at least

    for i in range(client_num - 1):
        train_data_dirichlet_list = np.append(train_data_dirichlet_list, int(N_available * proportions[i]))  # round down by inner function 'int'
    N_avaiable_left = N_available - np.sum(train_data_dirichlet_list)
    train_data_dirichlet_list = np.append(train_data_dirichlet_list, N_avaiable_left)

    train_data_dirichlet_list = train_data_dirichlet_list + np.array([1]*client_num)# add 1 sample to each client
    train_data_dirichlet_list_slice = []
    idx = 0
    for num in train_data_dirichlet_list:
        idx = idx + int(num)
        train_data_dirichlet_list_slice.append(idx)

    # Debug info
    # logging.info("train_data: {}".format(train_data))
    # logging.info("train_data_dirichlet_list: {}".format(train_data_dirichlet_list))
    # logging.info("train_data_dirichlet_list_slice: {}".format(train_data_dirichlet_list_slice))
    
    # generate the new train_data for each client
    train_data_dirichlet = np.split(train_data, train_data_dirichlet_list_slice)

    return train_data_dirichlet

def delete_cache(gen, output_dir):
    if gen > 4 :
        delete_model_path = os.path.join(output_dir, f'g{gen-3}')
        logging.info("Delete model cache {delete_model_path}".format(delete_model_path=delete_model_path))
        os.system('rm -rf {delete_model_path}'.format(delete_model_path=delete_model_path))
    else:
        pass

def seperate_clients(train_data, unlabeled_data, eval_data, beta, seed, client_num_in_total):
    client_num_in_total = client_num_in_total
    random.Random(seed).shuffle(train_data)
    random.Random(seed).shuffle(unlabeled_data) 
    random.Random(seed).shuffle(eval_data) # shuffle data for spliting

    train_data = np.array(train_data)
    unlabeled_data = np.array(unlabeled_data)
    eval_data = np.array(eval_data)

    if beta:
        train_data_sperate = partition_class_samples_with_dirichlet_distribution(train_data=train_data, beta=beta, client_num=client_num_in_total, seed=seed)
    else:
        train_data_sperate = np.split(train_data,client_num_in_total)
    unlabeled_data_seperate = np.split(unlabeled_data,client_num_in_total)
    eval_data_seperate = np.split(eval_data,client_num_in_total)
    return train_data_sperate, unlabeled_data_seperate, eval_data_seperate
def eval_softlabel(ipet_data, train_data, replace=False):
    if replace:
            logging.info("Correct button is on.")

    data_num = len(ipet_data)
    correct = 0
    for data in ipet_data:
        uid = data.guid
        
        true_label = None
        for labeled_data in train_data:
            if labeled_data.guid == uid:
                true_label = labeled_data.label
        if true_label == data.label:
            logging.info("Data {} is tagged correctly as {}.".format(uid, data.label))
            correct = correct + 1
        else:
            logging.info("Data {} is tagged wrong. Current label is {}, true label is ".format(uid, data.label ,true_label))
        
        if replace:
            data.label = true_label
    
    correct_ratio = correct / data_num
    logging.info("Inference correct ratio is {}".format(correct_ratio))

    return ipet_data

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


def train_ipet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
               ensemble_eval_config: EvalConfig, ipet_config: IPetConfig, final_model_config: WrapperConfig,
               final_train_config: TrainConfig, final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str,
               ensemble_repetitions: int = 3, final_repetitions: int = 1, reduction: str = 'wmean',
               train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
               eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True, seed: int = 42):
    """
    Train and evaluate a new iPET model for a given task.

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
    for gen in range(ipet_config.generations):
        gen_output_dir = os.path.join(output_dir, f'g{gen}')

        # Step 1: Train an ensemble of models corresponding to individual patterns
        ipet_data_dir = os.path.join(output_dir, f'g{gen - 1}', 'next-gen-train-data') if gen > 0 else None
        train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                           gen_output_dir, ipet_data_dir=ipet_data_dir,
                           repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                           eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=True)

        # Step 2: Use the model to annotate examples for the next generation
        original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
        num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data))
        generate_ipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                 labels=ensemble_model_config.label_list, logits_dir=gen_output_dir,
                                 output_dir=os.path.join(gen_output_dir, 'next-gen-train-data'), reduction=reduction,
                                 num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                 n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed)

    # Step 3: Merge the annotations created by each individual model
    logits_dir = os.path.join(output_dir, f'g{ipet_config.generations - 1}')
    logits_file = os.path.join(logits_dir, 'unlabeled_logits.txt')
    merge_logits(logits_dir, logits_file, reduction)
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logging.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 4: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval)


def train_pet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
              ensemble_eval_config: EvalConfig, final_model_config: WrapperConfig, final_train_config: TrainConfig,
              final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str, ensemble_repetitions: int = 3,
              final_repetitions: int = 1, reduction: str = 'wmean', train_data: List[InputExample] = None,
              unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None, do_train: bool = True,
              do_eval: bool = True, no_distillation: bool = False, seed: int = 42):
    """
    Train and evaluate a new PET model for a given task.

    :param ensemble_model_config: the model configuration for each model corresponding to an individual PVP
    :param ensemble_train_config: the training configuration for each model corresponding to an individual PVP
    :param ensemble_eval_config: the evaluation configuration for each model corresponding to an individual PVP
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
    :param no_distillation: if true, no distillation is performed
    :param seed: the random seed to use
    """

    # Step 1: Train an ensemble of models corresponding to individual patterns
    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids, output_dir,
                       repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=do_train, do_eval=do_eval,
                       save_unlabeled_logits=not no_distillation, seed=seed)

    if no_distillation:
        return

    # Step 2: Merge the annotations created by each individual model
    logits_file = os.path.join(output_dir, 'unlabeled_logits.txt')
    merge_logits(output_dir, logits_file, reduction)
    logits = LogitsList.load(logits_file).logits
    assert len(logits) == len(unlabeled_data)
    logging.info("Got {} logits from file {}".format(len(logits), logits_file))
    for example, example_logits in zip(unlabeled_data, logits):
        example.logits = example_logits

    # Step 3: Train the final sequence classifier model
    final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    final_train_config.use_logits = True

    train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
                     repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                     eval_data=eval_data, do_train=do_train, do_eval=do_eval, seed=seed)


def train_classifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42):
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

    train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[0], output_dir=output_dir,
                       repetitions=repetitions,
                       train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data, do_train=do_train,
                       do_eval=do_eval, seed=seed)

def train_fedpet(ensemble_model_config: WrapperConfig, ensemble_train_config: TrainConfig,
               ensemble_eval_config: EvalConfig, ipet_config: IPetConfig, final_model_config: WrapperConfig,
               final_train_config: TrainConfig, final_eval_config: EvalConfig, pattern_ids: List[int], output_dir: str,
               ensemble_repetitions: int = 3, final_repetitions: int = 1, reduction: str = 'wmean',
               train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
               eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True, seed: int = 42, aggregated: bool = True,
               augmentation: bool = True, fed: bool = True, vanilla: bool = True, beta: int = None, client_num_in_total: int = None):
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

    client_num_in_total = client_num_in_total
    num_clients = min(10, client_num_in_total) 
    train_data_sperate, unlabeled_data_seperate, eval_data_seperate = seperate_clients(train_data, unlabeled_data, eval_data, beta, seed, client_num_in_total)

    # Calculate the samples numbers of all clients involved in training
    sample_num_list = np.array([])
    for client in range(client_num_in_total):
        sample_num_list = np.append(sample_num_list, len(train_data_sperate[client]))
    logging.info("All clients: sample_num_list is {}".format(sample_num_list))

    if vanilla:
        for gen in range(ipet_config.generations):
            delete_cache(gen, output_dir)
            # Select clients
            np.random.seed(gen)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            logging.info("Gen {}: client_indexes is {}".format(gen, client_indexes))
            aggregated_model_path = None
            
            # Calculate the samples numbers of each clients involved in this round 
            sample_num_list = np.array([])
            for client in range(num_clients):
                sample_num_list = np.append(sample_num_list, len(train_data_sperate[client_indexes[client]]))
            logging.info("Gen{}: sample_num_list is {}".format(gen, sample_num_list))

            # Aggergate models trained in previous round.
            if gen > 0:
                aggregated_model_path = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
                models_path = []
                for i in range(num_clients):
                    pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')
                    models_path.append(pattern_iter_input_dir)
                models, fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                wrapper = init_model(ensemble_model_config)
                wrapper.model = fl_model

                logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
                wrapper.save(aggregated_model_path)   

            for client in range(num_clients):
                client_idx = client_indexes[client]
                gen_output_dir = os.path.join(output_dir, f'g{gen}', f'client{client}')
                train_data = train_data_sperate[client_idx].tolist()
                unlabeled_data = unlabeled_data_seperate[client_idx].tolist()
                eval_data = eval_data_seperate[client_idx].tolist()

                train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                                gen_output_dir,
                                repetitions=1, train_data=train_data, unlabeled_data=unlabeled_data,
                                eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=False,aggregated_model_path=aggregated_model_path)
    else:
        for gen in range(ipet_config.generations): # debug mode: start from 2nd iteration; defalut is 0
            delete_cache(gen, output_dir)
            # Select clients
            aggregated_model_path = None
            np.random.seed(gen)
            client_indexes = np.array([])
            if fed:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            else:
                client_indexes = range(10)
            
            logging.info("Gen {}: client_indexes is {}".format(gen, client_indexes))

            sample_num_list = np.array([])
            for client in range(num_clients):
                sample_num_list = np.append(sample_num_list, len(train_data_sperate[client_indexes[client]]))
            logging.info("Gen{}: sample_num_list is {}".format(gen, sample_num_list))

            for client in range(num_clients):
                # if client != 2: # debug mode
                #     continue
                client_idx = client_indexes[client]
                gen_output_dir = os.path.join(output_dir, f'g{gen}', f'client{client}')
                train_data = train_data_sperate[client_idx].tolist()
                unlabeled_data = unlabeled_data_seperate[client_idx].tolist()
                eval_data = eval_data_seperate[client_idx].tolist()

                if gen > 0:
                    if fed and aggregated: # 是否和其它方联合训练, fed avg 
                        models_path = []
                        aggregated_model_path = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
                        
                        for i in range(num_clients):
                            pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')
                            models_path.append(pattern_iter_input_dir)

                        models, fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                        wrapper = init_model(ensemble_model_config)
                        wrapper.model = fl_model
                        logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
                        wrapper.save(aggregated_model_path) 

                    if augmentation: # 是否利用unlabeled data
                        if fed: # 是否和其它方联合训练
                            # Step 1: Use all the model in last interation to annotate examples for this generation
                            if aggregated: # 是否把多方的模型聚合，以此来利用unlabeled data
                                models_path = []
                                aggregated_model_path = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
                                
                                for i in range(num_clients):
                                    pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')
                                    models_path.append(pattern_iter_input_dir)

                                models, fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                                wrapper = init_model(ensemble_model_config)
                                wrapper.model = fl_model
                                logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
                                wrapper.save(aggregated_model_path) 
                                logits = evaluate(wrapper, unlabeled_data, ensemble_eval_config)['logits']

                                save_logits(os.path.join(output_dir, f'g{gen-1}', f'client{0}', 'logits.txt'), logits)
                                

                                original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
                                # num_new_examples = 10 - len(train_data)
                                num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data)) # 由原先的**级别增长，降至*级别增长
                                generate_fedipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                                    labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                                    output_dir=os.path.join(output_dir, f'g{gen}', f'client{client}', 'this-gen-train-data'), reduction=reduction,
                                                    num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                                    n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated)

                            else: # 如果aggreagetd 没开，那就是每一方都会用其它n方client的模型来进行n次infer，而不是n方模型的进行
                                for i in range(num_clients):

                                    pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')

                                    wrapper = TransformerModelWrapper.from_pretrained(pattern_iter_input_dir)

                                    # evaluate on this client's unlabeled_data
                                    logits = evaluate(wrapper, unlabeled_data, ensemble_eval_config)['logits']

                                    # 本轮某个client的logits暂存在上一轮相应client的目录下，以供generate train sets使用; 会覆盖掉上一个client用自己的数据扩充的数据
                                    save_logits(os.path.join(pattern_iter_input_dir, 'logits.txt'), logits)

                                original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
                                num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data)) # 由原先的**级别增长，降至*级别增长
                                generate_fedipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                                    labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                                    output_dir=os.path.join(output_dir, f'g{gen}', f'client{client}', 'this-gen-train-data'), reduction=reduction,
                                                    num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                                    n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated)

                        else: # local pet
                            models_path = []
                                    
                            pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{client_idx}')
                            models_path.append(pattern_iter_input_dir)

                            models, fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
                            wrapper = init_model(ensemble_model_config)
                            wrapper.model = fl_model
                            logits = evaluate(wrapper, unlabeled_data, ensemble_eval_config)['logits']

                            save_logits(os.path.join(output_dir, f'g{gen-1}', f'client{0}', 'logits.txt'), logits)

                            original_data_size = len(train_data) if train_data else 10 / ipet_config.scale_factor
                            num_new_examples = int(original_data_size * (ipet_config.scale_factor ** (gen + 1)) - len(train_data)) # 由原先的**级别增长，降至*级别增长
                            generate_fedipet_train_sets(train_data=train_data, unlabeled_data=unlabeled_data,
                                                labels=ensemble_model_config.label_list, logits_dir=os.path.join(output_dir, f'g{gen-1}'),
                                                output_dir=os.path.join(output_dir, f'g{gen}', f'client{client}', 'this-gen-train-data'), reduction=reduction,
                                                num_new_examples=num_new_examples, logits_percentage=ipet_config.logits_percentage,
                                                n_most_likely=ipet_config.n_most_likely if gen == 0 else -1, seed=seed, aggregated=aggregated)


                # Step 2: Train an ensemble of models corresponding to individual clients (pattern = 1)
                ipet_data_dir = os.path.join(output_dir, f'g{gen}', f'client{client}', 'this-gen-train-data') if gen > 0 and augmentation else None
                last_iteration_model_path = None
                if gen > 0: # Inherit the model para. from last iteration
                    last_iteration_model_path = os.path.join(output_dir, f'g{gen-1}', f'client{client}')
                if aggregated and fed:
                    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                                    gen_output_dir, ipet_data_dir=ipet_data_dir,
                                    repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                                    eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=augmentation, aggregated_model_path = aggregated_model_path) 
                else:
                    train_pet_ensemble(ensemble_model_config, ensemble_train_config, ensemble_eval_config, pattern_ids,
                                    gen_output_dir, ipet_data_dir=ipet_data_dir,
                                    repetitions=ensemble_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                                    eval_data=eval_data, do_train=do_train, do_eval=do_eval, save_unlabeled_logits=augmentation, last_iteration_model_path = last_iteration_model_path) 


    # Step 3: Merge the annotations created by each individual model 这一步相当于generate_ipet_train_sets。但是是把所有的unlabeled data都打标签了。如果有多个pattern的话，在本地可以把多个pattern infer出来的logit聚合。现在我们只有一个patter，这一步无用。注：多个client infer出来的logit，由于隐私愿意，其原始数据和中间logits不可以传输。所以没法聚合。
    # logits_dir = os.path.join(output_dir, f'g{ipet_config.generations - 1}')
    # logits_file = os.path.join(logits_dir, 'unlabeled_logits.txt')
    # merge_logits(logits_dir, logits_file, reduction)
    # logits = LogitsList.load(logits_file).logits
    # assert len(logits) == len(unlabeled_data)
    # logging.info("Got {} logits from file {}".format(len(logits), logits_file))
    # for example, example_logits in zip(unlabeled_data, logits):
    #     example.logits = example_logits

    # Step 4: Train the final sequence classifier model 这一步会变成，让每个client单独训练一个sc，然后聚合sc模型。和常规的fl一样了。
    # final_model_config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
    # final_train_config.use_logits = True

    # train_classifier(final_model_config, final_train_config, final_eval_config, os.path.join(output_dir, 'final'),
    #                  repetitions=final_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
    #                  eval_data=eval_data, do_train=do_train, do_eval=do_eval)

def train_fedclassifier(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig, output_dir: str,
                     repetitions: int = 3, train_data: List[InputExample] = None,
                     unlabeled_data: List[InputExample] = None, eval_data: List[InputExample] = None,
                     do_train: bool = True, do_eval: bool = True, seed: int = 42, beta: int = None):
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
    client_num_in_total = client_num_in_total
    num_clients = min(10, client_num_in_total) 
    
    train_data_sperate, unlabeled_data_seperate, eval_data_seperate = seperate_clients(train_data, unlabeled_data, eval_data, beta, seed, client_num_in_total)

    for gen in range(repetitions):
        delete_cache(gen, output_dir)
        # Select clients
        np.random.seed(gen)
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("Gen {}: client_indexes is {}".format(gen, client_indexes))
        aggregated_model_path = None

        # Calculate the samples numbers of each clients involved in this round 
        sample_num_list = np.array([])
        for client in range(num_clients):
            sample_num_list = np.append(sample_num_list, len(train_data_sperate[client_indexes[client]]))
        logging.info("Gen{}: sample_num_list is {}".format(gen, sample_num_list))

        # Aggergate models trained in previous round.
        if gen > 0:
            aggregated_model_path = os.path.join(output_dir, f'g{gen-1}',f'aggregated')
            models_path = []
            for i in range(num_clients):
                pattern_iter_input_dir = os.path.join(output_dir, f'g{gen-1}',f'client{i}')
                models_path.append(pattern_iter_input_dir)
            models, fl_model = aggregate(models_path=models_path, sample_num_list=sample_num_list)
            wrapper = init_model(model_config)
            wrapper.model = fl_model

            logging.info("Saving aggregated trained model at {}...".format(aggregated_model_path))
            wrapper.save(aggregated_model_path)   

        for client in range(num_clients):
            client_idx = client_indexes[client]
            gen_output_dir = os.path.join(output_dir, f'g{gen}', f'client{client}')
            train_data = train_data_sperate[client_idx].tolist()
            unlabeled_data = unlabeled_data_seperate[client_idx].tolist()
            eval_data = eval_data_seperate[client_idx].tolist()

            train_pet_ensemble(model_config, train_config, eval_config, pattern_ids=[1], output_dir=gen_output_dir,
                            repetitions=1,
                            train_data=train_data, unlabeled_data=unlabeled_data, eval_data=eval_data, do_train=do_train,
                            do_eval=do_eval, seed=seed, aggregated_model_path=aggregated_model_path)

def train_pet_ensemble(model_config: WrapperConfig, train_config: TrainConfig, eval_config: EvalConfig,
                       pattern_ids: List[int], output_dir: str, ipet_data_dir: str = None, repetitions: int = 3,
                       train_data: List[InputExample] = None, unlabeled_data: List[InputExample] = None,
                       eval_data: List[InputExample] = None, do_train: bool = True, do_eval: bool = True,
                       save_unlabeled_logits: bool = False, seed: int = 42, aggregated_model_path: str = None, last_iteration_model_path: str = None):
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

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):

            model_config.pattern_id = pattern_id
            results_dict = {}

            fed = 1
            if fed:
                pattern_iter_output_dir = "{}".format(output_dir)
            else:
                pattern_iter_output_dir = "{}/p{}-i{}".format(output_dir, pattern_id, iteration)

            if os.path.exists(pattern_iter_output_dir) and do_train and not fed: # eval&fed的时候不要跳过
                logging.warning(f"Path {pattern_iter_output_dir} already exists, skipping it...")
                continue

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

                # Using trained model after 3 epochs of PET unsupervised learning (Only for SC testing, will be removed)
                inherit = 0
                if inherit: 
                    logging.info("Loading previous aggregated trained model via PET unsupervised learning.")
                    from transformers import BertForSequenceClassification
                    wrapper.model = BertForSequenceClassification.from_pretrained('./log_10_100/p0-i0')
                
                if ipet_data_dir:
                    p = os.path.join(ipet_data_dir, 'train.bin')
                    ipet_train_data = InputExample.load_examples(p)
                    for example in ipet_train_data:
                        example.logits = None
                    
                    logging.info("Evaluating soft label~")
                    ipet_train_data = eval_softlabel(ipet_train_data, unlabeled_data, replace=False)
                else:
                    ipet_train_data = None
                
                # # See test acc before training
                # eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)

                results_dict.update(train_single_model(wrapper, train_data, train_config, eval_config,
                                                       ipet_train_data=ipet_train_data,
                                                       unlabeled_data=unlabeled_data, eval_data = eval_data))

                with open(os.path.join(pattern_iter_output_dir, 'results.txt'), 'w') as fh:
                    fh.write(str(results_dict))

                logging.info("Saving trained model at {}...".format(pattern_iter_output_dir))
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

                # (Only for SC testing, will be removed)
                # wrapper.config = wrapper._load_config('./log_10_100/final/p0-i0')
                # wrapper.config.wrapper_type = SEQUENCE_CLASSIFIER_WRAPPER
                # from transformers import BertForSequenceClassification
                # wrapper.model = BertForSequenceClassification.from_pretrained(pattern_iter_output_dir)
                # from pet import preprocessor
                # wrapper.preprocessor = preprocessor.SequenceClassifierPreprocessor(
                #     wrapper, wrapper.config.task_name, 0, wrapper.config.verbalizer_file)
                # sc_eval_result = evaluate(wrapper, eval_data, eval_config, priming_data=train_data)
                # logging.info(sc_eval_result['scores']['acc'])

                save_predictions(os.path.join(pattern_iter_output_dir, 'predictions.jsonl'), wrapper, eval_result)
                save_logits(os.path.join(pattern_iter_output_dir, 'eval_logits.txt'), eval_result['logits'])

                scores = eval_result['scores']
                logging.info("--- RESULT (pattern_id={}, iteration={}) ---".format(pattern_id, iteration))
                logging.info(scores)
                logging.info(pattern_iter_output_dir)

                results_dict['test_set_after_training'] = scores
                with open(os.path.join(pattern_iter_output_dir, 'results.json'), 'w') as fh:
                    json.dump(results_dict, fh)

                for metric, value in scores.items():
                    results[metric][pattern_id].append(value)

                wrapper.model = None
                wrapper = None
                torch.cuda.empty_cache()

    if do_eval:
        logging.info("=== OVERALL RESULTS ===")
        _write_results(os.path.join(output_dir, 'result_test.txt'), results)
    else:
        logging.info("=== ENSEMBLE TRAINING COMPLETE ===")


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
        if train_data and return_train_set_results:
            results_dict['train_set_before_training'] = evaluate(model, eval_data, eval_config)['scores']['acc']

        
        logging.info('init acc: val acc before training is {}'.format(results_dict['train_set_before_training']))

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

    if train_data and return_train_set_results:
        results_dict['train_set_after_training'] = evaluate(model, train_data, eval_config)['scores']['acc']

    return results_dict


def evaluate(model: TransformerModelWrapper, eval_data: List[InputExample], config: EvalConfig,
             priming_data: List[InputExample] = None) -> Dict:
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


def _write_results(path: str, results: Dict):
    with open(path, 'w') as fh:
        for metric in results.keys():
            for pattern_id, values in results[metric].items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                result_str = "{}-p{}: {} +- {}".format(metric, pattern_id, mean, stdev)
                logging.info(result_str)
                fh.write(result_str + '\n')

        for metric in results.keys():
            all_results = [result for pattern_results in results[metric].values() for result in pattern_results]
            all_mean = statistics.mean(all_results)
            all_stdev = statistics.stdev(all_results) if len(all_results) > 1 else 0
            result_str = "{}-all-p: {} +- {}".format(metric, all_mean, all_stdev)
            logging.info(result_str)
            fh.write(result_str + '\n')


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
                             logits_percentage: float, n_most_likely: int = -1, seed: int = 42, aggregated: bool = True):
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
        if aggregated and subdir != 'client0':
            print(subdir,' ','no aggregated data')
            continue
        print(subdir,' ','has aggregated data')
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
        logits_lists[subdir] = loglist


    other_logits_lists = [ll for sd, ll in logits_lists.items()]
    subdir_train_set = generate_ipet_train_set(
        other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
        logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
        rng_np=rng_np
    )

    InputExample.save_examples(subdir_train_set,
                                os.path.join(output_dir, 'train.bin'))

def generate_ipet_train_sets(train_data: List[InputExample], unlabeled_data: List[InputExample], labels: List[str],
                             logits_dir: str, output_dir: str, reduction: str, num_new_examples: int,
                             logits_percentage: float, n_most_likely: int = -1, seed: int = 42):
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
        logits_lists[subdir] = loglist

    for subdir in subdirs:
        other_logits_lists = [ll for sd, ll in logits_lists.items() if sd != subdir]
        subdir_train_set = generate_ipet_train_set(
            other_logits_lists, labels=labels, original_data=unlabeled_data, examples_per_label=examples_per_label,
            logits_percentage=logits_percentage, reduction=reduction, n_most_likely=n_most_likely, rng=rng,
            rng_np=rng_np
        )

        InputExample.save_examples(subdir_train_set,
                                   os.path.join(output_dir, subdir + '-train.bin'))


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

    assert len(set(len(ll.logits) for ll in logits_lists)) == 1

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
        sum_label_probabilities = sum(label_probabilities)
        label_probabilities = [p / sum_label_probabilities for p in label_probabilities]
        return rng.choice(examples, size=num_examples, replace=False, p=label_probabilities).tolist()
    else:
        logging.info("This client has 0 examples to draw. Funtion _draw_examples_by_label_probability is passed.")
        return []

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def aggregate(models_path=None, sample_num_list=None):
    if not models_path:
        models_path = ['./log_10_100/p0-i0', 
        './log_10_100/p1-i0', 
        './log_10_100/p2-i0', 
        './log_10_100/p3-i0', 
        './log_10_100/p4-i0']
    models = []
    for model_path in models_path:
        model = TransformerModelWrapper.from_pretrained(model_path).model
        models.append(model)

    worker_state_dict =[x.state_dict()for x in models]
    weight_keys =list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum =0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key] * sample_num_list[i]
        fed_state_dict[key]= key_sum / np.sum(sample_num_list)
    #### update fed weights to fl model
    fl_model = TransformerModelWrapper.from_pretrained(models_path[0]).model
    fl_model.load_state_dict(fed_state_dict)
    return models, fl_model

def compare_model(m1, m2):
    worker_state_dict = [m1.state_dict(), m2.state_dict()]
    weight_keys =list(worker_state_dict[0].keys())
    import collections
    sub_state_dict = collections.OrderedDict() # subtraction state dictionary
    for key in weight_keys:
        sub_state_dict[key]= worker_state_dict[0][key] - worker_state_dict[1][key]
    sub_model = TransformerModelWrapper.from_pretrained('./log_10_100/p0-i0').model
    sub_model.load_state_dict(sub_state_dict)
    return sub_model
