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

from encodec import EncodecModel
from encodec import quantization as qt
from encodec.msstftd import MultiScaleSTFTDiscriminator
import encodec.modules as m

from VoiceExperiments.modules.losses import *

class EnCodec(EncodecModel):
    def __init__(self, model_config, optimizer_configs=None):
        params = model_config["params"]

        target_bandwidths = params["target_bandwidths"]
        sample_rate = params["sample_rate"]
        channels = params["channels"]
        causal = params["causal"]
        model_norm = params["model_norm"]
        audio_normalize = params["audio_normalize"]
        segment = params["segment"]
        name = params["name"]
        
        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)

        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=1024,
        )

        super(EnCodec, self).__init__(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )


        self.msstftd_descrim = MultiScaleSTFTDiscriminator(**model_config["MultiScaleSTFTDiscriminator"]) # TODO: need any special parsing of args


    def gen_step(self, x, lengths_x):
        frames = self.encode(x)
        G_x = self.decode(frames)[:, :, :x.shape[-1]]
        return G_x, None
        

    def train_step(self, batch):
        self.train()
        x, lengths_x = batch
        device = next(self.parameters()).device

        x = x.to(device)
        lengths_x = lengths_x.to(device)

        print(x.shape, self.sample_rate)
        G_x, commit_loss = self.gen_step(x, lengths_x)
        
        loss_g = 

        
    def eval_step(self, batch):
        pass

    def get_loss_g(self, generated, truth):
        # Losses
        # - Reconstruction Loss (time and frequency)
        # - Discriminative Loss
        # - VQ Commitment loss
        # - Balancer (not a loss, but a way of combining)
        # Also need to consider multi-bandwidth training
        #   note from paper: 
        #       (for Multi STFT Descrim) descriminator tend to overpower easily the decoder
        #       we update its weight with a probability of 2/3 at 24 kHz and 0.5 at 48 kHz

        pass