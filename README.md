# Federated Few-shot Learning for Mobile NLP (NFS)

NFS is an training-inference orchestration framework for enable private mobile NLP training with few labels.

NFS is built atop Pattern-Exploiting Training (PET) (commit id: 21d32d), the document file and instruction could be found in [README_pet.md](./README_pet.md)

# Step-by-step installation
After `git clone`-ing this repository, please run the following command to install our dependencies.

```bash
# create directly via conda (recommended)
conda env create -f environment.yaml
# or you can create the environment manually
conda create -n nfs python=3.7
pip install -r requirements.txt
```

## System requirement
Our system is implemented on:

 `Linux Phoenix22 5.4.0-122-generic #138~18.04.1-Ubuntu SMP Fri Jun 24 14:14:03 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux`

# Download data
```bash
# remebmer to modify the path in download.sh
bash script_shell/download.sh
```

# Main experiemnts
```bash
# RoBERTa-large
conda activate nfs
python sweep_aug.py
```

Above shells will run all datasets and models for three times (different seeds) automatically.The log files would be placed into `log/ablation`. 

# Reproduce main results in the paper
We process the result log via `figs/script`
to get the final pictures in the manuscript.

Our experiment results could be downloaded from [Google drive](https://drive.google.com/file/d/1HbNzjU8kk3PBihqEh-po2s7WGxL3ZIWE/view?usp=sharing).

# Customization
## 1. Manully run
First, you should turn off line 118 to `auto = False` in `sweep_aug.py`. Now you have following options to customize your experiments:
```bash
--dataset DATASET     Available datasets: agnews, mnli, yahoo, yelp-full
--method METHOD       Available methods: fedclassifier, fedpet
--device DEVICE       CUDA_VISIABLE_DEVICE
--train_examples TRAIN_EXAMPLES
                    done: 40; todo: 10, 100, 1000
--test_examples TEST_EXAMPLES
                    8700 for mnli
--unlabeled_examples UNLABELED_EXAMPLES
                    392700 for mnli
--alpha ALPHA         Data label similarity of each client, the larger the
                    beta the similar data for each client
--beta BETA           Int similarity of each client, the larger the beta the
                    similar data for each client. 0 for off
--gamma GAMMA         The labeled data distribution density, the larger the
                    gamma the uniform the labeled data distributed
--client_num_in_total CLIENT_NUM_IN_TOTAL
                    How many clients owe labeled data?
--all_client_num_in_total ALL_CLIENT_NUM_IN_TOTAL
                    How many clients are sperated
--pattern_ids PATTERN_IDS
                    pattern_ids
--seed SEED           seed
--model MODEL         model
--model_name_or_path MODEL_NAME_OR_PATH
                    model_name_or_path
--data_point DATA_POINT
                    How many data is to be annotated. Now, the increase
                    ratio of augment data
--conver_point CONVER_POINT
                    After conver_point, clients with unlabeled data will
                    be involved.
--limit LIMIT         logits < limit will be dropped
--num_clients_infer NUM_CLIENTS_INFER
                    select how many clients to do soft label annotation
--infer_freq INFER_FREQ
                    the model trains for infer_freq rounds, and annotation
                    starts once
--vote_k VOTE_K       whether to use vote_k. vote_k is the percentage of
                    unlabeled data for inferring
```
## 2. More experiments
Access [modeling.py](pet/modeling.py) line 58-68 to switch on/off key designs, e.g., filtering, bias-tuning, curriculum pacing, etc.
Remembert to modify the running shell [run_fed_aug.sh](run_fed_aug.sh) line 20-21 accordingly.