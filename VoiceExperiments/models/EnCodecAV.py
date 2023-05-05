# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation.
Adapted from https://github.com/facebookresearch/encodec/ for research purposes
"""


import typing as tp

import numpy as np
import torch
from torch import nn

from encodec import EncodecModel

class EnCodecAV(nn.Module):
    def __init__(self, ):


        super(EnCodecAV, self).__init__()

    
    def train_step(self, batch):
        pass

    def eval_step(self, batch):
        pass