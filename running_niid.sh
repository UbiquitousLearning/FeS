nohup python sweep.py --dataset agnews --device 1 --train_examples 120 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 --all_client_num_in_total 1000 --pattern_ids 5 --seed 42 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 120 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 8 --all_client_num_in_total 1000 --pattern_ids 1 --seed 42 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 650 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 650 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset yahoo --device 1 --train_examples 1400 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 5 &

nohup python sweep.py --dataset yahoo --device 2 --train_examples 1400 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &