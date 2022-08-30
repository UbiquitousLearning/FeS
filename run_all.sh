nohup python sweep.py --dataset agnews --device 0 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --beta 1 &

nohup python sweep.py --dataset agnews --device 1 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --beta 1 &

nohup python sweep.py --dataset mnli --device 1 --train_examples 100 --test_examples 8700 --unlabeled_examples 392700 --method fedclassifier --beta 1 &

nohup python sweep.py --dataset mnli --device 5 --train_examples 100 --test_examples 8700 --unlabeled_examples 392700 --method fedpet --beta 1 &

nohup python sweep.py --dataset yahoo --device 5 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --beta 1 &

nohup python sweep.py --dataset yahoo --device 7 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --beta 1 &

nohup python sweep.py --dataset yelp-full --device 4 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedclassifier --beta 1 &

nohup python sweep.py --dataset yelp-full --device 4 --train_examples 100 --test_examples -1 --unlabeled_examples -1 --method fedpet --beta 1 &