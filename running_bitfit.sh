# running: bitfit for four datasets. lr: 1e-3

seed=6

nohup python sweep.py --dataset agnews --device 4 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 100 --seed ${seed} --pattern_ids 1 --alpha 1 --gamma 0.001 &

nohup python sweep.py --dataset mnli --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 100 --seed ${seed} --pattern_ids 0 --alpha 0 --gamma 0.001 &

nohup python sweep.py --dataset yahoo --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --seed ${seed} --pattern_ids 5 --alpha 0 --gamma 0.001 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 100 --seed ${seed} --pattern_ids 0 --alpha 0 --gamma 0.001 &