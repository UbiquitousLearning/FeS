# # no check_data
# # a=1
# seed=42
# device=0
# data_point=5


# # nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# # nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &



# seed=42
# device=1
# data_point=10

# # nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# # nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &



# seed=42
# device=2
# data_point=20

# # nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# # nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &




# a=0 eval_iter=10
seed=42
device=4
data_point=5
alpha=0

nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &


seed=42
device=6
data_point=10
alpha=0

nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &


seed=42
device=7
data_point=20
alpha=0

nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha ${alpha} --data_point ${data_point} --num_clients_infer 5 --infer_freq 1 &