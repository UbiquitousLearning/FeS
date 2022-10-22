# running
# agnews datapoint = 0, a=1, g=0.001, clients = 32
nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 10 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 15 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 20 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 10 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 15 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 20 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 10 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 15 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 20 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 5 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 10 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 15 --infer_freq 1 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 20 --infer_freq 1 --model_name_or_path roberta-base &


# k = 2
nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 5 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 10 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 15 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 20 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 5 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 10 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 15 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 20 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 5 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 10 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 15 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 20 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 5 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 10 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 15 --infer_freq 2 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 20 --infer_freq 2 --model_name_or_path roberta-base &


# k = 4
nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 5 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 10 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 15 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 1 --num_clients_infer 20 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 5 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 10 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 15 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 5 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --num_clients_infer 20 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 5 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 10 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 15 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --num_clients_infer 20 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 5 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 10 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 15 --infer_freq 4 --model_name_or_path roberta-base &

nohup python sweep_aug.py --dataset agnews --device 6 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 8 --num_clients_infer 20 --infer_freq 4 --model_name_or_path roberta-base &