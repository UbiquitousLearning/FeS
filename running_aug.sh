nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 5 --conver_point 10 --limit 0 &

nohup python sweep_aug.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 --data_point 20 --conver_point 0 --limit 0 &

nohup python sweep.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma 0.001 --alpha 1 &