from typing import List, Dict
from pet.utils import InputExample, LogitsList, eq_div, softmax
import numpy as np
import os
import logging
import random
import ast
from copy import deepcopy

debug = False

process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

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