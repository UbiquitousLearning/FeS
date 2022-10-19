import logging 

import os
process_id = os.getpid()
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')


def deactivate_relevant_gradients(model):
        """
        https://github.com/benzakenelad/BitFit.git
        Turns off the model parameters requires_grad except the trainable_components.

        Args:
            model: from wrapper model. model.model is the common training model
            trainable_components (List[str]): list of trainable components (the rest will be deactivated)
            we only use bias (the baseline version of bitfit)

        BIAS_TERMS_DICT = {
        'intermediate': 'intermediate.dense.bias',
        'key': 'attention.self.key.bias',
        'query': 'attention.self.query.bias',
        'value': 'attention.self.value.bias',
        'output': 'output.dense.bias',
        'output_layernorm': 'output.LayerNorm.bias',
        'attention_layernorm': 'attention.output.LayerNorm.bias',
        'all': 'bias',
        }
        """
        logging.info(f"Before bitfit, model parameters is: {get_parameter_number(model.model)}")
        for param in model.model.parameters():
            param.requires_grad = False
        component = "bias"
        # trainable_components = trainable_components + ['classifier']
        for name, param in model.model.named_parameters():
            if component in name:
                logging.info(f"name {name}; param {param}")
                param.requires_grad = True
                # continue
        logging.info(f"After bitfit, model parameters is: {get_parameter_number(model.model)}")
        return model

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}