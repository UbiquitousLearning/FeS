gamma=0.001
# FedCLS thick
nohup python sweep.py --dataset mnli --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &

nohup python sweep.py --dataset yahoo --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &


# FedPrompt thick
nohup python sweep.py --dataset mnli --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &

nohup python sweep.py --dataset yahoo --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &

nohup python sweep.py --dataset yelp-full --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 &


# AUG-FedPrompt thick
nohup python sweep_aug.py --dataset mnli --device 0 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 10 --limit 0 &

nohup python sweep_aug.py --dataset yahoo --device 0 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 10 --limit 0 &

nohup python sweep_aug.py --dataset yahoo --device 3 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 1 --limit 0 &

nohup python sweep_aug.py --dataset yelp-full --device 0 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 3 --limit 0 &




# FedCLS thick
nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --gamma ${gamma} --alpha 1 &

# FedPrompt thick
nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --gamma ${gamma} --alpha 1 &

# AUG-FedPrompt thick
nohup python sweep_aug.py --dataset agnews --device 0 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 5 --limit 0.9 &

nohup python sweep_aug.py --dataset agnews --device 3 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 --gamma ${gamma} --alpha 1 --data_point 20 --conver_point 5 &