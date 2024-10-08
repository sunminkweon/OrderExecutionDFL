import torch
from model import *
from lightning_module import *
from utils import *

def load_model_parameters(model_name, config, file_path):
    # Load the checkpoint
    model_class = eval(model_name)
    model = model_class(config)
    model.load_state_dict(torch.load(file_path))
    
    if model_name == 'Seq2Seq':
        module_type = Seq2SeqModule.LightningSeq2Seq
    else:
        module_type = ModelModule.LigtningModel
        
    return model, module_type

def load_model_prediction( model, test_loader) : 
    pred = None
    target = None
    
    for x,y in test_loader :
        y_hat = model(x)
        
        if pred is None:
            pred = y_hat
            target = y
        else:
            # Stack each batch along the batch dimension (axis 0)
            pred = torch.cat((pred, y_hat), dim=0)
            target = torch.cat((target, y), dim=0)
    
    return pred, target
                