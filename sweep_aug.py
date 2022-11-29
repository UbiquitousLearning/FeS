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
    parser.add_argument('--method', type=str, default="fedpet",
                        help='Available methods: fedclassifier, fedpet')
    parser.add_argument('--device', type=int, default=1,
                        help='CUDA_VISIABLE_DEVICE')
    parser.add_argument('--train_examples', type=int, default=64,
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
    parser.add_argument("--client_num_in_total", type=int, default=32,
                        help="How many clients owe labeled data?")
    parser.add_argument("--all_client_num_in_total", type=int, default=1000,
                        help="How many clients are sperated")
    parser.add_argument("--pattern_ids", type=int, default=0,
                        help="pattern_ids")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed")
    parser.add_argument("--model", type=str, default="roberta",
                        help="model")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large",
                        help="model_name_or_path")
    parser.add_argument("--data_point", type=int, default=5,
                        help="How many data is to be annotated. Now, the increase ratio of augment data")
    parser.add_argument("--conver_point", type=int, default=0,
                        help="After conver_point, clients with unlabeled data will be involved.")
    parser.add_argument("--limit", type=float, default=0,
                        help="logits < limit will be dropped")
    parser.add_argument("--num_clients_infer", type=int, default=5,
                        help="select how many clients to do soft label annotation")
    parser.add_argument("--infer_freq", type=int, default=1,
                        help="the model trains for infer_freq rounds, and annotation starts once")
    parser.add_argument("--vote_k", type=float, default=0,
                        help="whether to use vote_k. vote_k is the percentage of unlabeled data for inferring")
                        
    return parser.parse_args()

def set_hp(dataset, method, device, train_examples, test_examples, unlabeled_examples, alpha, beta, gamma, client_num_in_total, all_client_num_in_total, pattern_ids, seed, model, model_name_or_path, data_point, conver_point, limit, num_clients_infer, infer_freq, vote_k):
    default = 0

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

    hp = dataset + " " + method + " " + str(device) + " " + str(train_examples) + " " + str(test_examples) + " " + str(unlabeled_examples) + " " + str(alpha) + " " + str(beta) + " " + str(gamma) + " " + str(client_num_in_total) + " " + str(all_client_num_in_total) + " " + str(pattern_ids) + " " + str(seed) + " " + str(model) + " " + str(model_name_or_path) + " " + str(data_point) + " " + str(conver_point) + " " + str(limit) + " " + str(num_clients_infer) + " " + str(infer_freq) + " " + str(vote_k)

    return hp


def set_hp_list(dataset, method, device, train_examples, test_examples, unlabeled_examples, alpha, beta, gamma, client_num_in_total, all_client_num_in_total, pattern_ids, seed, model, model_name_or_path, data_point, conver_point, limit, num_clients_infer, infer_freq, vote_k):

    hp = dataset + " " + method + " " + str(device) + " " + str(train_examples) + " " + str(test_examples) + " " + str(unlabeled_examples) + " " + str(alpha) + " " + str(beta) + " " + str(gamma) + " " + str(client_num_in_total) + " " + str(all_client_num_in_total) + " " + str(pattern_ids) + " " + str(seed) + " " + str(model) + " " + str(model_name_or_path) + " " + str(data_point) + " " + str(conver_point) + " " + str(limit) + " " + str(num_clients_infer) + " " + str(infer_freq) + " " + str(vote_k)

    return hp

# customize the log format
logging.basicConfig(level=logging.INFO,
                    format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')

parser = argparse.ArgumentParser()
args = add_args(parser)

auto = True

if not auto:
    args.hp = set_hp(args.dataset, args.method, args.device, args.train_examples, args.test_examples, args.unlabeled_examples, args.alpha, args.beta, args.gamma, args.client_num_in_total, args.all_client_num_in_total, args.pattern_ids, args.seed, args.model, args.model_name_or_path, args.data_point, args.conver_point, args.limit, args.num_clients_infer, args.infer_freq, args.vote_k)

    logging.info(args)
    logging.info('nohup bash run_fed_aug.sh '
                '{args.hp} '.format(args=args))
    os.system('nohup bash run_fed_aug.sh '
                '{args.hp} '.format(args=args))
else:
    # Fixed para.
    pattern_ids = {"agnews": 1, "yahoo": 5, "yelp-full": 0, "mnli": 0}
    all_client_num_in_total_list = {"agnews": 100, "yahoo": 1000, "yelp-full": 1000, "mnli": 1000}
    alphas = {"agnews": 1, "yahoo": 0, "yelp-full": 0, "mnli": 0}
    gammas = {"agnews": 0.001, "yahoo": 0.001, "yelp-full": 0.001, "mnli": 100}
    

    # Vary para.
    datasets = ['agnews', 'mnli', 'yahoo', 'yelp-full'] # 'agnews', 'mnli', 'yahoo', 'yelp-full'
    num_clients_infer_list = [5] # [1, 5, 10]
    infer_freq_list = [1]
    seeds = [42] 
    vote_k_list = [-1] # 0.01, 0.05, 0.1, 0.2
    vote_k_specific = None
    vote_k_specific = {"agnews": 0.1, "yahoo": 0.1, "yelp-full": 0.5, "mnli": 0.2} # this will cover vote_k_list
    datapoints = [5]
    models = ["roberta"] # "roberta", "bert", "albert", "roberta", "bert"
    model_name_or_path_list = ["roberta-large"] # "roberta-base", "bert-base-uncased", "albert-base-v2", "roberta-large", "bert-large-uncased"

    
    process = 0
    process_per_gpu = 4
    device_list = [3] # 0,1,2,3,4,5,6,7
    device_idx = 0


for num_clients_infer in num_clients_infer_list:
    args.num_clients_infer = num_clients_infer
    for infer_freq in infer_freq_list:
        args.infer_freq = infer_freq
        for datapoint in datapoints:
            args.data_point = datapoint
            for seed in seeds:
                args.seed = seed
                for vote_k in vote_k_list:
                    args.vote_k = vote_k
                    for dataset in datasets:
                        if vote_k_specific:
                            args.vote_k = vote_k_specific[dataset]
                        args.all_client_num_in_total = all_client_num_in_total_list[dataset]
                        args.dataset = dataset
                        args.pattern_ids = pattern_ids[dataset]
                        args.alpha = alphas[dataset]
                        args.gamma = gammas[dataset]
                        for idx in range(len(models)):
                            args.model = models[idx]
                            args.model_name_or_path = model_name_or_path_list[idx]
                            
                            args.device = device_list[device_idx]
                            args.hp = set_hp_list(args.dataset, args.method, args.device, args.train_examples, args.test_examples, args.unlabeled_examples, args.alpha, args.beta, args.gamma, args.client_num_in_total, args.all_client_num_in_total, args.pattern_ids, args.seed, args.model, args.model_name_or_path, args.data_point, args.conver_point, args.limit, args.num_clients_infer, args.infer_freq, args.vote_k)

                            logging.info(args)
                            logging.info('nohup bash run_fed_aug.sh &'
                                        '{args.hp} '.format(args=args))
                            os.system('nohup bash run_fed_aug.sh '
                                        '{args.hp} &'.format(args=args))
                            process += 1
                            if process >= process_per_gpu:
                                process = 0
                                device_idx += 1
