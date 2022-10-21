# running: bitfit for four datasets. lr: 1e-3

# skewed
# nohup python sweep.py --dataset mnli --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yahoo --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 5 --gamma 0.001 --model_name_or_path roberta-large &

# nohup python sweep.py --dataset yelp-full --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --model_name_or_path roberta-large &

# evenly
# nohup python sweep.py --dataset agnews --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 1 --gamma 100 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 100 --model_name_or_path roberta-large &

nohup python sweep.py --dataset yahoo --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 5 --gamma 100 --model_name_or_path roberta-large &

# nohup python sweep.py --dataset yelp-full --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 100 --model_name_or_path roberta-large &

# local
# nohup python sweep.py --dataset agnews --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 1 --model_name_or_path roberta-large &

nohup python sweep.py --dataset mnli --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &

# nohup python sweep.py --dataset yahoo --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 5 --model_name_or_path roberta-large &

# nohup python sweep.py --dataset yelp-full --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1000 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &

# lr = 1e-3 for no fitbit
# nohup python sweep.py --dataset mnli --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large &




# to run 

# nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 6 &

# nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 42 &

# nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 99 &

# nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 --pattern_ids 0 --model_name_or_path roberta-large --seed 42 &