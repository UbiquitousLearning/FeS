import argparse
import logging
import os
from time import sleep


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser.add_argument('--dataset', type=str, default="agnews",
                        help='Available datasets: agnews, mnli, yahoo, yelp-full')
    parser.add_argument('--method', type=str, default="fedclassifier",
                        help='Available methods: fedclassifier, fedpet')
    parser.add_argument('--device', type=int, default=1,
                        help='CUDA_VISIABLE_DEVICE')
    parser.add_argument('--train_examples', type=int, default=10,
                        help='done: 40; todo: 10, 100, 1000')
    parser.add_argument('--test_examples', type=int, default=-1,
                        help='8700 for mnli')
    parser.add_argument('--unlabeled_examples', type=int, default=-1,
                        help='392700 for mnli')
    parser.add_argument("--alpha", type=int, default=0,
                        help="Data label similarity of each client, the larger the beta the similar data for each client")
    parser.add_argument('--beta', type=int, default=0,
                        help='Int  similarity of each client, the larger the beta the similar data for each client. 0 for off')
    parser.add_argument("--gamma", type=float, default=0,
                        help="The labeled data distribution density, the larger the gamma the uniform the labeled data distributed")
    parser.add_argument("--client_num_in_total", type=int, default=10,
                        help="How many clients owe labeled data?")
    parser.add_argument("--all_client_num_in_total", type=int, default=100,
                        help="How many clients are sperated")
    parser.add_argument("--pattern_ids", type=int, default=0,
                        help="pattern_ids")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed")
    parser.add_argument("--model", type=str, default="roberta",
                        help="model")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large",
                        help="model_name_or_path")
                        
    return parser.parse_args()

def set_hp(dataset, method, device, train_examples, test_examples, unlabeled_examples, alpha, beta, gamma, client_num_in_total, all_client_num_in_total, pattern_ids, seed, model, model_name_or_path):
    default = False
    if default:
        dataset = 'yelp-full'
        method = "fedpet"
        pattern_ids = 0
        alpha=1
        beta=0 
        seed =99

        samples_per_client = int(train_examples / client_num_in_total)

        if samples_per_client == 1:
            train_examples = 32
        if samples_per_client == 2:
            train_examples = 64
        if samples_per_client == 4:
            train_examples = 128
        if samples_per_client == 8:
            train_examples = 256
        if samples_per_client == 16:
            train_examples = 512
        if samples_per_client == 32:
            train_examples = 1024

        if client_num_in_total == 1:
            gamma=0.001
        if client_num_in_total == 2:
            gamma=0.01
        if client_num_in_total == 4:
            gamma=0.1
        if client_num_in_total == 8:
            gamma=1
        if client_num_in_total == 16:
            gamma=10
        if client_num_in_total == 32:
            gamma=100
        
        client_num_in_total = 32

    if dataset == "agnews":
        all_client_num_in_total = 1000
    if dataset == "yahoo":
        all_client_num_in_total = 1000
    if dataset == "yelp-full":
        all_client_num_in_total = 1000
    if dataset == "boolq":
        all_client_num_in_total = 50
    if dataset == "mnli":
        all_client_num_in_total = 1000

    hp = dataset + " " + method + " " + str(device) + " " + str(train_examples) + " " + str(test_examples) + " " + str(unlabeled_examples) + " " + str(alpha) + " " + str(beta) + " " + str(gamma) + " " + str(client_num_in_total) + " " + str(all_client_num_in_total) + " " + str(pattern_ids) + " " + str(seed) + " " + str(model) + " " + str(model_name_or_path)

    return hp

# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)


args.hp = set_hp(args.dataset, args.method, args.device, args.train_examples, args.test_examples, args.unlabeled_examples, args.alpha, args.beta, args.gamma, args.client_num_in_total, args.all_client_num_in_total, args.pattern_ids, args.seed, args.model, args.model_name_or_path)

logging.info(args)
logging.info('nohup bash run_fed_aug.sh '
            '{args.hp} '.format(args=args))
os.system('nohup bash run_fed_aug.sh '
            '{args.hp} '.format(args=args))
