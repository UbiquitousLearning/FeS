gamma=0.001
seed=6
device=1

# nohup python sweep.py --dataset agnews --device 1 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --seed 6 --pattern_ids 1 --gamma 0.001 --alpha 1 &

# nohup python sweep.py --dataset agnews --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 6 --pattern_ids 1 --gamma 0.001 --alpha 1 &

nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 6 --pattern_ids 1 --gamma 0.001 --alpha 1 --data_point 100 --conver_point 0 &

nohup python sweep_aug.py --dataset agnews --device 4 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 1 --gamma 0.001 --alpha 1 --data_point 100 --conver_point 0 &
# AUG-FedPrompt thick
# nohup python sweep_aug.py --dataset agnews --device ${device} --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 1 --gamma ${gamma} --alpha 1 --data_point 100 --conver_point 0 &

nohup python sweep_aug.py --dataset mnli --device 0 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 6 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 20 --conver_point 10 --limit 0 &

nohup python sweep_aug.py --dataset yahoo --device 2 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 42 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 100 --conver_point 1 --limit 0 &

nohup python sweep_aug.py --dataset yelp-full --device 3 --train_examples 0 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed 6 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 20 --conver_point 3 --limit 0 &
