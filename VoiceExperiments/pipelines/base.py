import os
import torch
import torch.optim as optim
import yaml

def save_model(epoch, step, model, path):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epoch,
                'steps': step,
                'model_state_dict': model.state_dict(),
                }, f'{path}')

def get_optimizer(opt_config):
    type = opt_config["type"].lower()
    
    if type == 'adam':
        return optim.Adam
    
    raise NotImplementedError(f"Optimizer {type} not found")


class BasePipeline:

    # Models of pipeline with trainable parameters
    # Each model is saved with separate checkpoint by default
    models = None

    def __init__(self, pipeline_cfg, optimizer_cfgs, device=None):
        self.pipeline_cfg = pipeline_cfg
        self.optimizer_cfgs = optimizer_cfgs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_step(self, batch):
        raise NotImplementedError()

    def eval_step(self, batch):
        raise NotImplementedError()

    def train(self):
        for model in self.models.values():
            model.train()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def to(self, device):
        for model in self.models.values():
            model.to(device)
    
    def compile(self):
        self.models = {name: torch.compile(model) for name, model in self.models.items()}

    def save_models(self, root, save_name, epoch=None, step=None):
        for model_name, model in self.models.items():
            model_path = os.path.join(root, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
                
            save_model(epoch, step, model, os.path.join(model_path, f'{model_name}_{save_name}.ckpt'))