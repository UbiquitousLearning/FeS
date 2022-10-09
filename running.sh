# process clean
# kill -9 $(ps -ef | grep cli.py| awk '{print $2}')

# kill specific processes
# kill -9 $(ps -ef | grep "cli.py.*--client_num_in_total 5"| awk '{print $2}')

# kill -9 $(ps -ef | grep "cli.py.*--method fedpet"| awk '{print $2}')

# kill -9 $(ps -ef | grep "cli.py.*--client_num_in_total 160"| awk '{print $2}')

# client_num_in_total_list=(1) # todo 10 50 100
# samples_per_client_list=(1 5 10 50 100)
# for client_num_in_total in $client_num_in_total_list
# do 
#     for samples_per_client in $samples_per_client_list
#     do
#         echo client_num_in_total: $client_num_in_total, samples_per_client: $samples_per_client is running...
#         train_examples=`expr $client_num_in_total \* $samples_per_client`
        
#         nohup python sweep.py --dataset agnews --device 6 --train_examples $train_examples --test_examples -1 --unlabeled_examples -1 --method fedpet --beta 1 --client_num_in_total $client_num_in_total &
#     done
# done


# client_num_in_total = 100
# samples_per_client_list = [1, 5, 10, 50, 100]
# nohup python sweep.py --dataset agnews --device 4 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

# Dataset: Agnews
# seed: 42


# fedcls

# -----------------------
# client_num_in_total: 1
# nohup python sweep.py --dataset agnews --device 1 --train_examples 1 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 2 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 1 &
# -----------------------
# client_num_in_total: 2
# nohup python sweep.py --dataset agnews --device 2 --train_examples 2 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 2 &

# -----------------------
# client_num_in_total: 4
# nohup python sweep.py --dataset agnews --device 1 --train_examples 4 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 &
# -----------------------
# client_num_in_total: 8
# nohup python sweep.py --dataset agnews --device 2 --train_examples 8 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 8 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 8 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 8 &


# # client_num_in_total: 16
# nohup python sweep.py --dataset agnews --device 2 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 16 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 16 &
# nohup python sweep.py --dataset agnews --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 16 &

# # -----------------------
# # client_num_in_total: 32
# nohup python sweep.py --dataset agnews --device 1 --train_examples 32 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 &
# nohup python sweep.py --dataset agnews --device 1 --train_examples 128 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 &


# nohup python sweep.py --dataset yelp-full --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &

# nohup python sweep.py --dataset yelp-full --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 &

# nohup python sweep.py --dataset wic --device 2 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 &

# nohup python sweep.py --dataset boolq --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 50 &

# nohup python sweep.py --dataset wic --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1 &


# nohup python sweep.py --dataset wic --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1000 &

# nohup python sweep.py --dataset mnli --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 &

# nohup python sweep.py --dataset yahoo --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 1 &

# nohup python sweep.py --dataset yelp-full --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

# nohup python sweep.py --dataset copa --device 7 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 1 --all_client_num_in_total 1 --pattern_ids 0 &

# nohup python sweep.py --dataset mnli --device 7 --train_examples 8192 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 &

# nohup python sweep.py --dataset mnli --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 &




# nohup python sweep.py --dataset yahoo --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 &

# nohup python sweep.py --dataset yahoo --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 &

# nohup python sweep.py --dataset yelp-full --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 &

# nohup python sweep.py --dataset boolq --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 50 &

# nohup python sweep.py --dataset mnli --device 7 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 32 --all_client_num_in_total 1000 &

# wait for the nvidia driver installed

# sleep 2h
# fedcls 4 * 4
nohup python sweep.py --dataset yahoo --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 --all_client_num_in_total 1000 &

nohup python sweep.py --dataset yelp-full --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset boolq --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 --all_client_num_in_total 50 &

nohup python sweep.py --dataset mnli --device 1 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 4 --all_client_num_in_total 1000 &


# fedpet 8 * 8
nohup python sweep.py --dataset yahoo --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 --all_client_num_in_total 1000 &

nohup python sweep.py --dataset yelp-full --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset boolq --device 2 --train_examples 512 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 50 &

nohup python sweep.py --dataset mnli --device 2 --train_examples 64 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 8 --all_client_num_in_total 1000 &

# fedpet 4 * 4
nohup python sweep.py --dataset yahoo --device 7 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 --all_client_num_in_total 1000 &

nohup python sweep.py --dataset yelp-full --device 7 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 --all_client_num_in_total 1000 --pattern_ids 0 &

nohup python sweep.py --dataset boolq --device 7 --train_examples 256 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 50 &

nohup python sweep.py --dataset mnli --device 7 --train_examples 16 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 4 --all_client_num_in_total 1000 &


nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 8 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 4 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 2 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 1 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.1 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.2 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.4 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.8 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.01 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.02 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.04 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 1024 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --client_num_in_total 50 --all_client_num_in_total 50 --pattern_ids 1 --alpha 1 --gamma 0.08 &


nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --data_point 20 &

nohup python sweep_aug.py --dataset mnli --device 1 --train_examples 392 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 32 --all_client_num_in_total 1000 --pattern_ids 0 --data_point 30 &