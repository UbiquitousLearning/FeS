# Dataset: Agnews
# seed: 42

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



# client_num_in_total = 10
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset agnews --device 1 --train_examples 10 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 50 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &

nohup python sweep.py --dataset agnews --device 5 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &

nohup python sweep.py --dataset agnews --device 5 --train_examples 1000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &


# client_num_in_total = 50
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset agnews --device 6 --train_examples 50 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 50 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 250 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 50 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 50 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 2500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 50 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 5000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 50 &


# client_num_in_total = 100
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset agnews --device 4 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 1000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 5000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 10000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &