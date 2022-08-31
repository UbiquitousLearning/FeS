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
    parser.add_argument('--beta', type=int, default=0,
                        help='Int  similarity of each client, the larger the beta the similar data for each client. 0 for off')
    parser.add_argument("--client_num_in_total", type=int, default=10,
                        help="How many clients owe labeled data?")
                        
    return parser.parse_args()

def set_hp(dataset, method, device, train_examples, test_examples, unlabeled_examples, beta, client_num_in_total):
        hp = dataset + " " + method + " " + str(device) + " " + str(train_examples) + " " + str(test_examples) + " " + str(unlabeled_examples) + " " + str(beta) + " " + str(client_num_in_total)

        return hp

# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)


# conda activate ptpretrain

# agnews 100
# bash run_fed.sh agnews fedpet 7 100 -1 -1
# bash run_fed.sh agnews fedclassifier 7 100 -1 -1

# mnli 100
# bash run_fed.sh mnli fedpet 7 100 8700 392700
# bash run_fed.sh mnli fedclassifier 7 100 8700 392700

# yahoo 100
# bash run_fed.sh yahoo fedpet 1 100 -1 -1
# bash run_fed.sh yahoo fedclassifier 1 100 -1 -1

# yelp-full 40
# bash run_fed.sh yelp-full fedpet 1 40 -1 -1
# bash run_fed.sh yelp-full fedclassifier 1 40 -1 -1

args.hp = set_hp(args.dataset, args.method, args.device, args.train_examples, args.test_examples, args.unlabeled_examples, args.beta, args.client_num_in_total)

logging.info("hp = %s" % args.hp)
os.system('nohup bash run_fed.sh '
            '{args.hp} '.format(args=args))
