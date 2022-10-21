# running
# yelp-full datapoint = 256, a=1, g=0.001, clients = 32
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 0 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 4 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 8 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 16 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 32 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 64 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

nohup python sweep_aug.py --dataset yelp-full --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 0 --data_point 100 --conver_point 0 --limit 0 --model_name_or_path roberta-large &


# # done


# # agnews datapoint = 64, a=1, g=0.001, clients = 32
# nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 2 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

# nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 4 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

# nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 16 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

# nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 32 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

# nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 64 --conver_point 0 --limit 0 --model_name_or_path roberta-large &

# nohup python sweep_aug.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --alpha 1 --gamma 0.001 --pattern_ids 1 --data_point 100 --conver_point 0 --limit 0 --model_name_or_path roberta-large &



