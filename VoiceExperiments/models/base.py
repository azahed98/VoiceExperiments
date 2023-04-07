import torch
import torch.optim as optim
from torch import nn
# from easydict import EasyDict

def get_optimizer(opt_config):
    type = opt_config["type"].lower()
    
    if type == 'adam':
        return optim.Adam
    
    raise NotImplementedError(f"Optimizer {type} not found")

class BaseModel(nn.Module):
    """
        Abstract class for defining models
        Models consist of parameters, any supporting modules, and optimizers (to allow separate training steps)
        Models implement the individual training step in such a way that training can just be looped over the dataloader
    """
    def __init__(self, model_config, opt_configs):
        pass
