import torch
import pytorch_lightning as pl

from FineVC.models import VCModel
# from FineVC.modules.encoders import ...
from torch import nn
from easydict import EasyDict


class FineVCModel(VCModel):
    def __init__(self, config):
        super(FineVCModel, self).__init__()

        # self.shared_encoder = ...

        # self.pitch_encoder = ...
        # self.amplitude_encoder = ...
        # self.speaker_encoder = ...
        # self.phoneme_encoder = ... # includes tokenizer
        # self.style_encoder = ...

        # self.decoder = ...


    def forward(self, args):
        
        # Extract pitch

        # Extract amplitude

        # Extract speaker

        # Extract phoneme

        # Extract style

        pass
    
    def _get_loss(self, args):

        pass
    
