import logging

import numpy as np
import math
import random

from pet.wrapper import TransformerModelWrapper

import os
process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

def find_labeled(labeled_idx, train_data, unlabeled_data, eval_data):
    # labeled_idx = []
    # for i in range(len(train_data)):
    #     data = train_data[i]
    #     if len(data) != 0:
    #         labeled_idx.append(i)
    # labeled_idx= np.array(labeled_idx)

    train_data_sperate = []
    unlabeled_data_seperate = []
    eval_data_seperate = []
    for idx in labeled_idx:
        train_data_sperate.append(train_data[idx] if idx < len(train_data) else [])
        unlabeled_data_seperate.append(unlabeled_data[idx])
        eval_data_seperate.append(eval_data[idx])
    train_data_sperate = np.array(train_data_sperate)
    unlabeled_data_seperate = np.array(unlabeled_data_seperate)
    eval_data_seperate = np.array(eval_data_seperate)
    return train_data_sperate, unlabeled_data_seperate, eval_data_seperate

    
def eval_softlabel(ipet_data, train_data, replace=False, labels = None):
    if labels: # ensemble voting
        logging.info("Ensemble voting is on.")

        pattern_ids = [0]
        data_num = len(ipet_data)
        correct = 0
        find_correct = 0
        find_wrong = 0
        ipet_data_all_right = []
        for j in range(len(ipet_data)):
            flag = 0 # 0 for wrong, 1 for right 
            data = ipet_data[j]
            uid = data.guid
            
            true_label = None
            for labeled_data in train_data:
                if labeled_data.guid == uid:
                    true_label = labeled_data.label
            if true_label == data.label:
                logging.info(f"Data {uid} is tagged correctly as {data.label}. Logits is {data.logits}")
                flag = 1
                correct = correct + 1
            else:
                logging.info(f"Data {uid} is tagged wrong. Current label is {data.label}, true label is {true_label}. Logits is {data.logits}")

             # correctly find those data annotated wrong
            for i in range(len(pattern_ids)):
                pattern_id = pattern_ids[i]
                
                if labels[i][j] == data.label:
                    logging.info(f"Data {uid} is tagged the same as p-{pattern_id}. Logits is {data.logits}")
                    
                else:
                    logging.info(f"Data {uid} is tagged differently. Current label is {data.label}, p-{pattern_id} label is {labels[i][j]}. Logits is {data.logits}")
                    if flag:
                        find_wrong = find_wrong + 1
                    else:
                        find_correct = find_correct + 1
                    break

            ipet_data_all_right.append(data)
        
        correct_ratio = correct / data_num
        find_correct_ratio = find_correct / data_num
        find_wrong_ratio = find_wrong / data_num
        
        logging.info(f"Inference correct ratio is {correct_ratio * 100}% ")
        logging.info(f"Ensemble voting finds {find_correct_ratio * 100}% correctly")
        logging.info(f"Ensemble voting finds {find_wrong_ratio * 100}% wrong")

        return ipet_data_all_right
    
    else:
        if replace:
            logging.info("Correct button is on.")

        data_num = len(ipet_data)
        correct = 0

        limit = 0
        ipet_data_within_limit = []
        for data in ipet_data:
            uid = data.guid
            
            true_label = None
            for labeled_data in train_data:
                if labeled_data.guid == uid:
                    true_label = labeled_data.label
            if true_label == data.label:
                logging.info(f"Data {uid} is tagged correctly as {data.label}. Logits is {data.logits}")
                correct = correct + 1
            else:
                logging.info(f"Data {uid} is tagged wrong. Current label is {data.label}, true label is {true_label}. Logits is {data.logits}")
                pass
                
            if data.logits[0] > limit:
                ipet_data_within_limit.append(data)
            
            if replace:
                data.label = true_label
        
        correct_ratio = correct / data_num
        logging.info(f"Inference correct ratio is {correct_ratio}")

        data_num = len(ipet_data_within_limit)
        correct = 0
        if data_num > 0: # replace and 
            for data in ipet_data_within_limit:
                uid = data.guid
                
                true_label = None
                for labeled_data in train_data:
                    if labeled_data.guid == uid:
                        true_label = labeled_data.label
                if true_label == data.label:
                    logging.info("Data {} is tagged correctly as {}.".format(uid, data.label))
                    correct = correct + 1
                else:
                    logging.info("Data {} is tagged wrong. Current label is {}, true label is {}".format(uid, data.label ,true_label))
                    pass
                
                if replace:
                    data.label = true_label
            
            correct_ratio = correct / data_num
            logging.info("After correct: Inference correct ratio is {}".format(correct_ratio))

        return ipet_data_within_limit

def get_prediction_accuracy_distribution(predictions, labels, label_list):
    labels_num = len(label_list)
    label_list = range(labels_num)
    # logging.info(f"label_list: {label_list}")
    train_examples_per_label = [sum(1 for i in range(len(predictions)) if labels[i] == label) for label in label_list]
    correct_per_label = [sum(1 for i in range(len(predictions)) if predictions[i] == labels[i] and labels[i] == label) for label in label_list]
    wrong_per_label = [sum(1 for i in range(len(predictions)) if predictions[i] != labels[i] and labels[i] == label) for label in label_list]
    ratio_per_label = []

    logging.info(f"Example distribution in the original dataset: {train_examples_per_label}")
    logging.info(f"correct_per_label: {correct_per_label}")
    logging.info(f"wrong_per_label: {wrong_per_label}")

    for i in range(labels_num):
        if train_examples_per_label[i] == 0:
            ratio_per_label.append(0)
        else:
            ratio_per_label.append(correct_per_label[i] / train_examples_per_label[i])


    logging.info(f"ratio_per_label: {ratio_per_label}")