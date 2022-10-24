# AGNEWS pattern: 1, alpha=1, gamma=0.001
nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-large &


# MNLI pattern: 0
nohup python sweep_aug.py --dataset mnli --device 4 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 1 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-large &


# YAHOO pattern: 0
nohup python sweep_aug.py --dataset yahoo --device 4 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 1 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-large &

# Yelp-full pattern: 0
nohup python sweep_aug.py --dataset yelp-full --device 4 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 1 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-large &