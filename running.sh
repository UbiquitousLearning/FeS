# process clean
# kill -9 $(ps -ef | grep cli.py| awk '{print $2}')

# kill specific processes
# kill -9 $(ps -ef | grep "cli.py.*--client_num_in_total 5 &"| awk '{print $2}')

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
nohup python sweep.py --dataset agnews --device 6 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 6 --train_examples 500 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 1000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 5000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &

nohup python sweep.py --dataset agnews --device 4 --train_examples 10000 --test_examples -1 --unlabeled_examples -1 --method fedpet --client_num_in_total 100 &


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