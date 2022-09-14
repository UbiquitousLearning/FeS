import logging

import numpy as np
import math
import random

import os
process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')


def get_examples_distribution(train_data, labels,state=0):
    
    train_examples_per_label = [sum(1 for ex in train_data if ex.label == label) for label in labels]
    if state == 1: # print origin client distribution
        logging.info(f"Origin client distribution: Example distribution in the original dataset: {train_examples_per_label}")
    elif state == 2: # print labeled client distribution
        logging.info(f"Labeled client distribution: Example distribution in the original dataset: {train_examples_per_label}")
    else: # common
        logging.info(f"Common client distribution: Example distribution in the original dataset: {train_examples_per_label}")


# label non-iid (alpha)
def dynamic_batch_fill(label_index_tracker, label_index_matrix,
                       remaining_length, current_label_id):
    """
    copied from fednlp
    params
    ------------------------------------------------------------------------
    label_index_tracker : 1d numpy array track how many data each label has used 
    label_index_matrix : 2d array list of indexs of each label
    remaining_length : int remaining empty space in current partition client list
    current_label_id : int current round label id
    ------------------------------------------------------------------------

    return 
    ---------------------------------------------------------
    label_index_offset: dict  dictionary key is label id 
    and value is the offset associated with this key
    ----------------------------------------------------------
    """
    remaining_unfiled = remaining_length
    label_index_offset = {}
    label_remain_length_dict = {}
    total_label_remain_length = 0
    # calculate total number of all the remaing labels and each label's remaining length
    for label_id, label_list in enumerate(label_index_matrix):
        if label_id == current_label_id:
            label_remain_length_dict[label_id] = 0
            continue
        label_remaining_count = len(label_list) - label_index_tracker[label_id]
        if label_remaining_count > 0:
            total_label_remain_length = (total_label_remain_length +
                                         label_remaining_count)
        else:
            label_remaining_count = 0
        label_remain_length_dict[label_id] = label_remaining_count
    length_pointer = remaining_unfiled

    if total_label_remain_length > 0:
        label_sorted_by_length = {
            k: v
            for k, v in sorted(label_remain_length_dict.items(),
                               key=lambda item: item[1])
        }
    else:
        label_index_offset = label_remain_length_dict
        return label_index_offset
    # for each label calculate the offset move forward by distribution of remaining labels
    for label_id in label_sorted_by_length.keys():
        fill_count = math.ceil(label_remain_length_dict[label_id] /
                               total_label_remain_length * remaining_length)
        fill_count = min(fill_count, label_remain_length_dict[label_id])
        offset_forward = fill_count
        # if left room not enough for all offset set it to 0
        if length_pointer - offset_forward <= 0 and length_pointer > 0:
            label_index_offset[label_id] = length_pointer
            length_pointer = 0
            break
        else:
            length_pointer -= offset_forward
            label_remain_length_dict[label_id] -= offset_forward
        label_index_offset[label_id] = offset_forward

    # still has some room unfilled
    if length_pointer > 0:
        for label_id in label_sorted_by_length.keys():
            # make sure no infinite loop happens
            fill_count = math.ceil(label_sorted_by_length[label_id] /
                                   total_label_remain_length * length_pointer)
            fill_count = min(fill_count, label_remain_length_dict[label_id])
            offset_forward = fill_count
            if length_pointer - offset_forward <= 0 and length_pointer > 0:
                label_index_offset[label_id] += length_pointer
                length_pointer = 0
                break
            else:
                length_pointer -= offset_forward
                label_remain_length_dict[label_id] -= offset_forward
            label_index_offset[label_id] += offset_forward

    return label_index_offset


def label_skew_process(train_data, labels, client_num, alpha, seed=42):
    """
    params
    -------------------------------------------------------------------
    labels : generate label_vocab
    train_data : generated label_assignment and data_length
    label_vocab : dict label vocabulary of the dataset 
    label_assignment : 1d list a list of label, the index of list is the index associated to label
    client_num : int number of clients
    alpha : float similarity of each client, the larger the alpha the similar data for each client
    -------------------------------------------------------------------
    return 
    ------------------------------------------------------------------
    partition_result : 2d array list of partition index of each client 
    ------------------------------------------------------------------
    """
    np.random.seed(seed)
    label_vocab = {}
    for label in labels:
        label_vocab[label] = labels.index(label)
    label_vocab = dict.keys(label_vocab)
    label_assignment = np.array([ex.label for ex in train_data])
    data_length = len(train_data)
    # print("label_vocab", label_vocab)
    # print("label_assignment", label_assignment, "len", len(label_assignment))
    # print(data_length)
    label_index_matrix = [[] for _ in label_vocab]
    label_proportion = []
    partition_result = [[] for _ in range(client_num)]
    client_length = 0
    # print("client_num", client_num)
    # shuffle indexs and calculate each label proportion of the dataset
    for index, value in enumerate(label_vocab):
        label_location = np.where(label_assignment == value)[0]
        label_proportion.append(len(label_location) / data_length)
        np.random.shuffle(label_location)
        label_index_matrix[index].extend(label_location[:])
    # print("proportion",label_proportion)
    # calculate size for each partition client
    label_index_tracker = np.zeros(len(label_vocab), dtype=int)
    total_index = data_length
    each_client_index_length = int(total_index / client_num)
    # print("each index length", each_client_index_length)
    client_dir_dis = np.array([alpha * l for l in label_proportion])
    # print("alpha", alpha)
    # print("client dir dis", client_dir_dis)
    proportions = np.random.dirichlet(client_dir_dis)
    # print("dir distribution", proportions)
    # add all the unused data to the client
    for client_id in range(len(partition_result)):
        each_client_partition_result = partition_result[client_id]
        proportions = np.random.dirichlet(client_dir_dis)
        # print(client_id,proportions)
        # print(type(proportions[0]))
        while True in np.isnan(proportions):
            proportions = np.random.dirichlet(client_dir_dis)
        client_length = min(each_client_index_length, total_index)
        if total_index < client_length * 2:
            client_length = total_index
        total_index -= client_length
        client_length_pointer = client_length
        # for each label calculate the offset length assigned to by Dir distribution and then extend assignment
        for label_id, _ in enumerate(label_vocab):
            offset = round(proportions[label_id] * client_length)
            if offset >= client_length_pointer:
                offset = client_length_pointer
                client_length_pointer = 0
            else:
                if label_id == (len(label_vocab) - 1):
                    offset = client_length_pointer
                client_length_pointer -= offset

            start = int(label_index_tracker[label_id])
            end = int(label_index_tracker[label_id] + offset)
            label_data_length = len(label_index_matrix[label_id])
            # if the the label is assigned to a offset length that is more than what its remaining length
            if end > label_data_length:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:])
                label_index_tracker[label_id] = label_data_length
                label_index_offset = dynamic_batch_fill(
                    label_index_tracker, label_index_matrix,
                    end - label_data_length, label_id)
                for fill_label_id in label_index_offset.keys():
                    start = label_index_tracker[fill_label_id]
                    end = (label_index_tracker[fill_label_id] +
                           label_index_offset[fill_label_id])
                    each_client_partition_result.extend(
                        label_index_matrix[fill_label_id][start:end])
                    label_index_tracker[fill_label_id] = (
                        label_index_tracker[fill_label_id] +
                        label_index_offset[fill_label_id])
            else:
                each_client_partition_result.extend(
                    label_index_matrix[label_id][start:end])
                label_index_tracker[
                    label_id] = label_index_tracker[label_id] + offset

        # if last client still has empty rooms, fill empty rooms with the rest of the unused data
        if client_id == len(partition_result) - 1:
            # print("last id length", len(each_client_partition_result))
            # print("Last client fill the rest of the unfilled lables.")
            for not_fillall_label_id in range(len(label_vocab)):
                if label_index_tracker[not_fillall_label_id] < len(
                        label_index_matrix[not_fillall_label_id]):
                    # print("fill more id", not_fillall_label_id)
                    start = label_index_tracker[not_fillall_label_id]
                    each_client_partition_result.extend(
                        label_index_matrix[not_fillall_label_id][start:])
                    label_index_tracker[not_fillall_label_id] = len(
                        label_index_matrix[not_fillall_label_id])
        partition_result[client_id] = each_client_partition_result
    
    train_data_seperated = []
    
    for partition in partition_result:
        client_data = []
        for id in partition:
            client_data.append(train_data[id])
        train_data_seperated.append(client_data)

    return train_data_seperated


def partition_class_samples_with_dirichlet_distribution(
        train_data, beta, client_num, seed):
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
    if beta:
        proportions = np.random.dirichlet(np.repeat(beta, client_num))
    else: # unifrom split
        proportions = np.array([1 / client_num] * client_num)

    train_data_dirichlet_list = np.array([])

    N_available = N - client_num # ensure that each client has 1 samples at least

    for i in range(client_num - 1):
        train_data_dirichlet_list = np.append(train_data_dirichlet_list, int(N_available * proportions[i]))  # round down by inner function 'int'
    N_avaiable_left = N_available - np.sum(train_data_dirichlet_list)
    train_data_dirichlet_list = np.append(train_data_dirichlet_list, N_avaiable_left)

    train_data_dirichlet_list = train_data_dirichlet_list + np.array([1]*client_num)# add 1 sample to each client
    train_data_dirichlet_list_slice = []
    idx = 0
    for num in train_data_dirichlet_list[:-1]:
        idx = idx + int(num)
        train_data_dirichlet_list_slice.append(idx)

    # Debug info
    # logging.info("train_data: {}".format(train_data))
    # logging.info("train_data_dirichlet_list: {}".format(train_data_dirichlet_list))
    # logging.info("train_data_dirichlet_list_slice: {}".format(train_data_dirichlet_list_slice))
    
    # generate the new train_data for each client
    train_data_dirichlet = np.split(train_data, train_data_dirichlet_list_slice)

    return train_data_dirichlet


def tag(train_and_unlabeled_data_sperate, client_num_in_total, all_client_num_in_total, train_examples, gamma, seed):
    train_data_sperate = []
    unlabeled_data_seperate = []

    np.random.seed(seed)
    for data in train_and_unlabeled_data_sperate:
        random.Random(seed).shuffle(data)

    if gamma:
        proportions = np.random.dirichlet(np.repeat(gamma, client_num_in_total))
    else: # unifrom split
        proportions = np.array([1 / client_num_in_total] * client_num_in_total)

    train_data_dirichlet_list = np.array([])

    N_available = train_examples - client_num_in_total # ensure that each client has 1 samples at least

    for i in range(client_num_in_total-1):
        train_data_dirichlet_list = np.append(train_data_dirichlet_list, int(N_available * proportions[i]))  # round down by inner function 'int'
    N_avaiable_left = N_available - np.sum(train_data_dirichlet_list)
    train_data_dirichlet_list = np.append(train_data_dirichlet_list, N_avaiable_left)

    train_data_dirichlet_list = train_data_dirichlet_list + np.array([1]*client_num_in_total)# add 1 sample to each client

    logging.info("len_labeled_data_list: {}".format(train_data_dirichlet_list))

    for i in range(client_num_in_total):
        offset = int(train_data_dirichlet_list[i])
        train_data_sperate.append(train_and_unlabeled_data_sperate[i][:offset])
        unlabeled_data_seperate.append(train_and_unlabeled_data_sperate[i][offset:])
    
    for i in range(all_client_num_in_total - client_num_in_total):
        unlabeled_data_seperate.append(train_and_unlabeled_data_sperate[i + client_num_in_total])
    
    train_data_sperate = np.array(train_data_sperate)
    unlabeled_data_seperate = np.array(unlabeled_data_seperate)

    return train_data_sperate, unlabeled_data_seperate


def seperate_clients(train_and_unlabeled_data_sperate, eval_data, alpha, beta, gamma, seed, client_num_in_total, all_client_num_in_total, train_examples, labels):
    global_seed = 42 # fix this will stable the foundation/common (1st layer) client partition
    if alpha:
        train_and_unlabeled_data_sperate = label_skew_process(train_data=train_and_unlabeled_data_sperate, labels=labels, client_num=all_client_num_in_total, alpha=alpha, seed=global_seed)
        for data in train_and_unlabeled_data_sperate:
            get_examples_distribution(data, labels, 1)
        eval_data_seperate = partition_class_samples_with_dirichlet_distribution(train_data=eval_data, beta=beta, client_num=all_client_num_in_total, seed=global_seed)
    
    else: # if beta = 0, will be uniformly distributed
        train_and_unlabeled_data_sperate = partition_class_samples_with_dirichlet_distribution(train_data=train_and_unlabeled_data_sperate, beta=beta, client_num=all_client_num_in_total, seed=global_seed) # dataset partition is fixed except for different niid.
        for data in train_and_unlabeled_data_sperate:
            get_examples_distribution(data, labels, 1)
        eval_data_seperate = partition_class_samples_with_dirichlet_distribution(train_data=eval_data, beta=beta, client_num=all_client_num_in_total, seed=global_seed)

    if seed != 42: # vary those clients with labels; 42 is the default, others need to be shuffled
        random.Random(seed).shuffle(train_and_unlabeled_data_sperate)

    train_data_sperate, unlabeled_data_seperate = tag(train_and_unlabeled_data_sperate, client_num_in_total, all_client_num_in_total, train_examples, gamma, seed)
    
    

    return train_data_sperate, unlabeled_data_seperate, eval_data_seperate
