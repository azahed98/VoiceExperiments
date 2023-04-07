import torch
import os
# from yaml import Dump
from torch.utils.tensorboard import SummaryWriter

def save_model(epochs, model, path):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                }, f'{path}')
    
class TensorBoardLogger:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, logging_cfg, best_valid_loss=float('inf'), model_cfg=None, training_cfg=None):
        self.best_valid_loss = best_valid_loss

        self.root = logging_cfg["root"]
        self.monitor = logging_cfg["monitor"]
        self.save_every_n_epochs = logging_cfg["save_every_n_epochs"]
        self.save_samples = logging_cfg["save_samples"]
        self.scalars = logging_cfg["scalars"]
        if "samples" in logging_cfg:
            self.sample_saving = logging_cfg["samples"]

        if not os.path.isdir(os.path.join(self.root, "checkpoints")):
            os.makedirs(os.path.join(self.root, "checkpoints"))
        if self.save_samples:
            # TODO
            raise NotImplementedError("Cannot save samples yet")
            os.makedirs(os.path.join(self.root, "samples"))
        self.writer = SummaryWriter(self.root)

    def log_train(self, results, epoch, step, model):
        for scalar in self.scalars:
            self.writer.add_scalar(f"train/{scalar}", results[scalar], step)


    def log_val(self, results, epoch, step, model):
        for scalar in self.scalars:
            self.writer.add_scalar(f"val/{scalar}", results[scalar], step)

        current_valid_loss = results[self.monitor]
        
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            path = os.path.join(self.root, "checkpoints/best.ckpt")
            save_model(epoch, model, path)
        
        if epoch % self.save_every_n_epochs == 0:
            path = os.path.join(self.root, "checkpoints/last.ckpt")
            save_model(epoch, model, path)
        
        