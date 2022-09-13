# Dataset: Agnews
# seed: 42

# -----------------------
# client_num_in_total: 1
nohup python sweep.py --dataset agnews --device 1 --train_examples 1 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 2 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 &
# -----------------------
# client_num_in_total: 2
nohup python sweep.py --dataset agnews --device 1 --train_examples 2 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 2 &

# -----------------------
# client_num_in_total: 4
nohup python sweep.py --dataset agnews --device 1 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 &
# -----------------------
# client_num_in_total: 8
nohup python sweep.py --dataset agnews --device 1 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 &


# -----------------------
# client_num_in_total: 16
nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 512 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 16 &
# -----------------------
# client_num_in_total: 32
nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &
nohup python sweep.py --dataset agnews --device 2 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &
nohup python sweep.py --dataset agnews --device 5 --train_examples 512 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &
nohup python sweep.py --dataset agnews --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &

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


# -----------------------
# client_num_in_total: 10
nohup python sweep.py --dataset agnews --device 3 --train_examples 10 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 20 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset agnews --device 6 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &

# -----------------------
# client_num_in_total: 20
nohup python sweep.py --dataset agnews --device 1 --train_examples 20 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset agnews --device 7 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset agnews --device 5 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset agnews --device 5 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &

# -----------------------
# client_num_in_total: 40
nohup python sweep.py --dataset agnews --device 0 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset agnews --device 5 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
# -----------------------
# client_num_in_total: 80
nohup python sweep.py --dataset agnews --device 3 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset agnews --device 3 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &

# -----------------------
# client_num_in_total: 160
nohup python sweep.py --dataset agnews --device 0 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset agnews --device 1 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset agnews --device 7 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset agnews --device 5 --train_examples 5120 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &

# -----------------------
# client_num_in_total: 320
nohup python sweep.py --dataset agnews --device 7 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset agnews --device 7 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset agnews --device 7 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset agnews --device 6 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 5120 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset agnews --device 4 --train_examples 10240 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &





# Dataset: Yelp-full
# seed: 42

# client_num_in_total = 100
# samples_per_client_list = [1, 5, 10, 50, 100]
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 1000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset yelp-full --device 4 --train_examples 5000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset yelp-full --device 4 --train_examples 10000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &


# -----------------------
# client_num_in_total: 10
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 10 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 20 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 10 &
# -----------------------
# client_num_in_total: 20
nohup python sweep.py --dataset yelp-full --device 3 --train_examples 20 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset yelp-full --device 4 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset yelp-full --device 5 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset yelp-full --device 7 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 20 &

# -----------------------
# client_num_in_total: 40
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 40 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 40 &
# -----------------------
# client_num_in_total: 80
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 80 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 80 &
# -----------------------
# client_num_in_total: 160
nohup python sweep.py --dataset yelp-full --device 6 --train_examples 160 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset yelp-full --device 7 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset yelp-full --device 7 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset yelp-full --device 1 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 5120 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 160 &
# -----------------------
# client_num_in_total: 320
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 320 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 640 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset yelp-full --device 2 --train_examples 1280 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset yelp-full --device 4 --train_examples 2560 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset yelp-full --device 5 --train_examples 5120 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &
nohup python sweep.py --dataset yelp-full --device 5 --train_examples 10240 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 320 &