# client_num_in_total = 1
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset agnews --device 6 --train_examples 1 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 5 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 10 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 50 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &


# client_num_in_total = 5
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset agnews --device 6 --train_examples 5 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 5 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 25 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 5 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 50 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 5 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 250 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 5 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 5 &