from pet.utils import InputExample, exact_match, save_logits, save_predictions, softmax, LogitsList, set_seed, eq_div
import os

ipet_data_dir = "/data2/cdq/pet/log_fedpet_augmentation/g1/client0/this-gen-train-data"
p = os.path.join(ipet_data_dir, 'train.bin')
ipet_train_data = InputExample.load_examples(p)



print(ipet_train_data)