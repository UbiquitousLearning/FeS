nohup python sweep.py --dataset mnli --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --gamma 0.001 --alpha 1 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --alpha 1 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 1 --train_examples 2048 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 1 --train_examples 4096 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &





nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --data_point 10 --conver_point 10 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --data_point 20 --conver_point 10 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 3 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 3 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 10 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 3 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 20 --conver_point 10 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 20 --conver_point 0 --limit 0 --model_name_or_path roberta-large &


nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 4 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 8 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 16 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 32 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 64 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 100 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 256 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 256 --conver_point 0 --limit 0 --model_name_or_path roberta-large &


nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 50 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 100 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset agnews --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --seed 6 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yahoo --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --seed 6 --gamma 0.001 --alpha 1 --data_point 10 --conver_point 0 --limit 0 --model_name_or_path roberta-large &




# running: bitfit for four datasets. lr: 1e-3

# skewed
nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 1 --gamma 0.001 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 5 --gamma 0.001 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --model_name_or_path roberta-large &

# evenly
nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 1 --gamma 100 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 100 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 5 --gamma 100 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 100 --model_name_or_path roberta-large &


# to run 

nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 6 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 42 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 99 &

# nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 42 &