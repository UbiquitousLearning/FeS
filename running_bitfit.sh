# running: bitfit for four datasets. lr: 1e-3

seed=42
# skewed
# nohup python sweep_aug.py --dataset mnli --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 0  --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

# nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 1 --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset mnli --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --alpha 0  --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &


# mannual running
# yahoo
nohup python sweep_aug.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 5 --alpha 1  --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 5 --alpha 1  --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 5 --alpha 0  --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 5 --alpha 0  --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &


# yelp-full
nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 0 --alpha 1  --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 0 --alpha 1  --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 0 --alpha 0  --gamma 100 --data_point 5 --num_clients_infer 5 --infer_freq 1 &

nohup python sweep_aug.py --dataset yelp-full --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 0 --alpha 0  --gamma 0.001 --data_point 5 --num_clients_infer 5 --infer_freq 1 &