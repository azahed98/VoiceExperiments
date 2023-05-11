# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation.
Adapted from https://github.com/facebookresearch/encodec/ for research purposes
"""

import math
import torch

import typing as tp

import numpy as np
from torch import nn

from torch import autocast
from torch.cuda.amp import GradScaler

from encodec import EncodecModel
from encodec import quantization as qt
from encodec.model import EncodedFrame
from encodec.msstftd import MultiScaleSTFTDiscriminator
import encodec.modules as m

from VoiceExperiments.pipelines.base import BasePipeline
from VoiceExperiments.modules.EnCodec import EnCodecGenerativeLoss, adversarial_d_loss
from VoiceExperiments.models.base import get_optimizer

class EnCodec(BasePipeline):
    def __init__(self, pipeline_cfg, optimizer_cfgs=None):
        super(EnCodec, self).__init__(pipeline_cfg, optimizer_cfgs)
        params = pipeline_cfg.params

        target_bandwidths = params.target_bandwidths
        sample_rate = params.sample_rate
        self.sample_rate = sample_rate
        channels = params.channels
        causal = params.causal
        model_norm = params.model_norm
        audio_normalize = params.audio_normalize
        segment = params.segment
        name = params.name
        
        # encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        # decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)

        # n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        # quantizer = qt.ResidualVectorQuantizer(
        #     dimension=encoder.dimension,
        #     n_q=n_q,
        #     bins=1024,
        # )

        # self.encodec = EncodecModel(
        #     encoder,
        #     decoder,
        #     quantizer,
        #     target_bandwidths,
        #     sample_rate,
        #     channels,
        #     normalize=audio_normalize,
        #     segment=segment,
        #     name=name,
        # )


        self.msstftd_descrim = MultiScaleSTFTDiscriminator(**pipeline_cfg.MultiScaleSTFTDiscriminator) # TODO: need any special parsing of args
        
        # # TODO: Add layer at end of descrim for aggregating across freq domain
        # # self.num_stft_descrim = len(self.msstftd_descrim.n_ffts) # number of sub-descrims
        # # self.dmsstftd_descrim_fc = nn.ModuleList([
        # #     nn.Linear()
        # # ])

        # self.gen_loss_fn = EnCodecGenerativeLoss(self.sample_rate)

        # self.optimizer_g = get_optimizer(optimizer_cfgs.gen)(
        #     list(self.encodec.parameters()),
        #     **optimizer_cfgs.gen.kwargs
        # )

        self.optimizer_d = get_optimizer(optimizer_cfgs.descrim)(
            list(self.msstftd_descrim.parameters()),
            **optimizer_cfgs.descrim.kwargs
        )
        
        self.models = {
            # "EnCodec" : self.encodec,
            # "Descriminator" : self.msstftd_descrim
        }
    
    
    def gen_step(self, x, lengths_x):
        pass
        

    def train_step(self, batch):
        self.train()
        x, lengths_x = batch
        device = self.device

        x = x.to(device)
        lengths_x = lengths_x.to(device)

        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()

        losses = {}

        return losses
        
    def eval_step(self, batch):
        self.eval()
        x, lengths_x = batch
        device = self.device

        x = x.to(device)
        lengths_x = lengths_x.to(device)

        losses = {}
        return losses