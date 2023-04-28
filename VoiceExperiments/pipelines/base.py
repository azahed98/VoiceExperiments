import os
import torch
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

class BasePipeline:

    # Models of pipeline with trainable parameters
    # Each model is saved with separate checkpoint by default
    models = None
    model_cfgs = None

    def __init__(self, pipeline_cfg, optimizer_cfgs):
        self.pipeline_cfg = pipeline_cfg
        self.optimizer_cfgs = optimizer_cfgs

    def train_step(self, batch):
        raise NotImplementedError()

    def eval_step(self, batch):
        raise NotImplementedError()

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def to(self, device):
        for model in self.models:
            model.to(device)
        
    def compile(self):
        self.models = {name: torch.compile(model) for name, model in self.models.items()}

    def save_models(self, root, save_name, epoch=None, step=None):
        for model_name, model in self.models.items():
            model_cfg = self.model_cfgs[model_name]
            model_path = os.path.join(root, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
                
            save_model(epoch, step, model, os.path.join(model_path, f'{model_name}_{save_name}.ckpt'))