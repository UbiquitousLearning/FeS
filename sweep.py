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
    parser.add_argument("--gamma", type=int, default=0,
                        help="The labeled data distribution density, the larger the gamma the uniform the labeled data distributed")
    parser.add_argument("--client_num_in_total", type=int, default=10,
                        help="How many clients owe labeled data?")
    parser.add_argument("--all_client_num_in_total", type=int, default=100,
                        help="How many clients are sperated")
    parser.add_argument("--pattern_ids", type=int, default=0,
                        help="pattern_ids")
                        
    return parser.parse_args()

def set_hp(dataset, method, device, train_examples, test_examples, unlabeled_examples, alpha, beta, gamma, client_num_in_total, all_client_num_in_total, pattern_ids):

    hp = dataset + " " + method + " " + str(device) + " " + str(train_examples) + " " + str(test_examples) + " " + str(unlabeled_examples) + " " + str(alpha) + " " + str(beta) + " " + str(gamma) + " " + str(client_num_in_total) + " " + str(all_client_num_in_total) + " " + str(pattern_ids)

    return hp

# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)


args.hp = set_hp(args.dataset, args.method, args.device, args.train_examples, args.test_examples, args.unlabeled_examples, args.alpha, args.beta, args.gamma, args.client_num_in_total, args.all_client_num_in_total, args.pattern_ids)

logging.info(args)
logging.info('nohup bash run_fed.sh '
            '{args.hp} '.format(args=args))
os.system('nohup bash run_fed.sh '
            '{args.hp} '.format(args=args))
