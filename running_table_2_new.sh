gamma=0.001
seed=6
device=1

# AUG-FedPrompt thick
# nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --gamma ${gamma} --alpha 1 --data_point 100 --conver_point 0 &

nohup python sweep_aug.py --dataset mnli --device ${device} --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 10 --limit 0 &

# nohup python sweep_aug.py --dataset yahoo --device ${device} --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 1 --limit 0 &

nohup python sweep_aug.py --dataset yelp-full --device ${device} --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 3 --limit 0 &
