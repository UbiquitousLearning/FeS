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



output_dir="/data/cdq/pet_data"
data_dir="${output_dir}/data"

dataset=$1 # agnews, mnli, yahoo, yelp-full
method=$2 # fedpet, fedclassifer
device=$3
train_examples=$4 # done: 40; todo: 10, 100, 1000, 
test_examples=$5 # 8700 for mnli
unlabeled_examples=$6 # 392700 for mnli
alpha=$7
beta=$8 # niid button, 0 for off.
gamma=$9
client_num_in_total=${10}
all_client_num_in_total=${11}
pattern_ids=${12}
seed=${13}
model_type=${14}
model_name_or_path=${15}

task_name=${dataset}

# pattern_ids=[1,0,2]
echo $pattern_ids
# model_type="roberta"
# model_name_or_path="roberta-base"

# model_type="roberta"
# model_name_or_path="roberta-large"

# model_type="bert"
# model_name_or_path="bert-base-uncased"

# model_type="albert"
# model_name_or_path="albert-base-v2"

# model_type="albert"
# model_name_or_path="albert-base-v2"

epochs=1
iteration=1000
clients=${client_num_in_total}

output_model_dir=/${output_dir}/log/${dataset}/all_${all_client_num_in_total}/seed_${seed}/pattern_${pattern_ids}/alpha_${alpha}_beta_${beta}_gamma_${gamma}
output_log_dir=./log/${dataset}/all_${all_client_num_in_total}/seed_${seed}/pattern_${pattern_ids}/alpha_${alpha}_beta_${beta}_gamma_${gamma}
mkdir -p $output_model_dir
mkdir -p $output_log_dir

echo $method "start."
CUDA_VISIBLE_DEVICES=$device python3 cli.py \
--train_examples ${train_examples} \
--test_examples ${test_examples} \
--unlabeled_examples ${unlabeled_examples} \
--method ${method} \
--pattern_ids ${pattern_ids} \
--data_dir $data_dir/${dataset} \
--model_type ${model_type} \
--model_name_or_path ${model_name_or_path} \
--task_name ${task_name} \
--output_dir $output_model_dir/all_aug_100_val_${method}_${train_examples}_${clients}_${model_name_or_path} \
--ipet_scale_factor 1 \
--ipet_generations ${iteration} \
--pet_num_train_epochs ${epochs} \
--pet_repetitions 1 \
--pet_per_gpu_train_batch_size 4 \
--sc_num_train_epochs ${epochs} \
--sc_repetitions ${iteration} \
--client_num_in_total ${client_num_in_total} \
--all_client_num_in_total ${all_client_num_in_total} \
--alpha ${alpha} \
--beta ${beta} \
--gamma ${gamma} \
--do_train \
--do_eval \
--fed \
--augmentation \
--seed ${seed} \
--aggregated > ${output_log_dir}/all_aug_100_val_${method}_${train_examples}_${clients}_${model_name_or_path}.log 2>&1
