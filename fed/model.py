from pet.wrapper import TransformerModelWrapper
import collections
import numpy as np
import gc

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def aggregate(models_path=None, sample_num_list=None):
    if not models_path:
        models_path = ['./log_10_100/p0-i0', 
        './log_10_100/p1-i0', 
        './log_10_100/p2-i0', 
        './log_10_100/p3-i0', 
        './log_10_100/p4-i0']
    models = []
    for model_path in models_path:
        model = TransformerModelWrapper.from_pretrained(model_path).model
        models.append(model)

    worker_state_dict =[x.state_dict() for x in models]
    weight_keys =list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum =0
        for i in range(len(models)):
            key_sum = key_sum + worker_state_dict[i][key] * sample_num_list[i]
        fed_state_dict[key]= key_sum / np.sum(sample_num_list)
    
    del models
    gc.collect()

    #### update fed weights to fl model
    fl_model = TransformerModelWrapper.from_pretrained(models_path[0]).model
    fl_model.load_state_dict(fed_state_dict)

    
    return fl_model

def compare_model(m1, m2):
    worker_state_dict = [m1.state_dict(), m2.state_dict()]
    weight_keys =list(worker_state_dict[0].keys())
    import collections
    sub_state_dict = collections.OrderedDict() # subtraction state dictionary
    for key in weight_keys:
        sub_state_dict[key]= worker_state_dict[0][key] - worker_state_dict[1][key]
    sub_model = TransformerModelWrapper.from_pretrained('./log_10_100/p0-i0').model
    sub_model.load_state_dict(sub_state_dict)
    return sub_model