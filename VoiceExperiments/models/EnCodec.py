# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# """EnCodec model implementation.
# Adapted from https://github.com/facebookresearch/encodec/ for research purposes
# """

# import math
# import torch

# import typing as tp

# import numpy as np
# from torch import nn

# from encodec import EncodecModel
# from encodec import quantization as qt
# from encodec.model import EncodedFrame
# from encodec.msstftd import MultiScaleSTFTDiscriminator
# import encodec.modules as m

# from VoiceExperiments.modules.EnCodec import EnCodecGenerativeLoss, adversarial_d_loss
# from VoiceExperiments.models.base import get_optimizer

# class EnCodec(EncodecModel):
#     def __init__(self, model_config, optimizer_configs=None):
#         params = model_config["params"]

#         target_bandwidths = params["target_bandwidths"]
#         sample_rate = params["sample_rate"]
#         channels = params["channels"]
#         causal = params["causal"]
#         model_norm = params["model_norm"]
#         audio_normalize = params["audio_normalize"]
#         segment = params["segment"]
#         name = params["name"]
        
#         encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal)
#         decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal)

#         n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))
#         quantizer = qt.ResidualVectorQuantizer(
#             dimension=encoder.dimension,
#             n_q=n_q,
#             bins=1024,
#         )

#         super(EnCodec, self).__init__(
#             encoder,
#             decoder,
#             quantizer,
#             target_bandwidths,
#             sample_rate,
#             channels,
#             normalize=audio_normalize,
#             segment=segment,
#             name=name,
#         )


#         self.msstftd_descrim = MultiScaleSTFTDiscriminator(**model_config["MultiScaleSTFTDiscriminator"]) # TODO: need any special parsing of args
        
#         # TODO: Add layer at end of descrim for aggregating across freq domain
#         # self.num_stft_descrim = len(self.msstftd_descrim.n_ffts) # number of sub-descrims
#         # self.dmsstftd_descrim_fc = nn.ModuleList([
#         #     nn.Linear()
#         # ])

#         self.gen_loss_fn = EnCodecGenerativeLoss(self.sample_rate)

#         self.optimizer_g = get_optimizer(optimizer_configs["gen"])(
#             list(self.encoder.parameters()) + list(self.quantizer.parameters()) + list(self.decoder.parameters()),
#             **optimizer_configs["gen"]["kwargs"]
#         )

#         self.optimizer_d = get_optimizer(optimizer_configs["descrim"])(
#             list(self.msstftd_descrim.parameters()),
#             **optimizer_configs["descrim"]["kwargs"]
#         )
        
#     def encode_verbose(self, x: torch.Tensor) -> tp.Tuple[tp.List[EncodedFrame], torch.Tensor]:
#         """Identital to encode but uses forward of the quantizer instead of 
#         encode, so as to get commitment loss for training
#         """
#         assert x.dim() == 3
#         _, channels, length = x.shape
#         assert channels > 0 and channels <= 2
#         segment_length = self.segment_length
#         if segment_length is None:
#             segment_length = length
#             stride = length
#         else:
#             stride = self.segment_stride  # type: ignore
#             assert stride is not None

#         encoded_frames: tp.List[EncodedFrame] = []
#         commit_losses = []
#         for offset in range(0, length, stride):
#             frame = x[:, :, offset: offset + segment_length]
#             quantized_result, scale = self._encode_frame_verbose(frame)
#             encoded_frames.append([quantized_result.codes, scale])
#             commit_losses.append(quantized_result.penalty)

#         return encoded_frames, torch.mean(torch.stack(commit_losses))
    
#     def _encode_frame_verbose(self, x: torch.Tensor) -> EncodedFrame:
#         length = x.shape[-1]
#         duration = length / self.sample_rate
#         assert self.segment is None or duration <= 1e-5 + self.segment

#         if self.normalize:
#             mono = x.mean(dim=1, keepdim=True)
#             volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
#             scale = 1e-8 + volume
#             x = x / scale
#             scale = scale.view(-1, 1)
#         else:
#             scale = None

#         emb = self.encoder(x)
#         quantized_result = self.quantizer(emb, self.frame_rate, self.bandwidth)
#         quantized_result.codes = quantized_result.codes.transpose(0,1)
#         # codes is [B, K, T], with T frames, K nb of codebooks.
#         return quantized_result, scale
    
#     def gen_step(self, x, lengths_x):
#         frames, commit_loss = self.encode_verbose(x)
#         G_x = self.decode(frames)[:, :, :x.shape[-1]]
#         return G_x, commit_loss
        

#     def train_step(self, batch):
#         self.train()
#         x, lengths_x = batch
#         device = next(self.parameters()).device

#         x = x.to(device)
#         lengths_x = lengths_x.to(device)

#         self.optimizer_g.zero_grad()
#         self.optimizer_d.zero_grad()

#         # Generate
#         G_x, commit_loss = self.gen_step(x, lengths_x)
        
#         # Discriminator
#         logits_x, features_stft_disc_x = self.msstftd_descrim(x)
#         logits_G_x, features_stft_disc_G_x = self.msstftd_descrim(G_x)
        
#         # Gen Loss
#         losses = self.gen_loss_fn.backward(
#             x, 
#             G_x, 
#             logits_G_x, 
#             features_stft_disc_x, 
#             features_stft_disc_G_x, 
#             commit_loss,
#             list(self.parameters())
#         )
#         self.optimizer_g.step()
        
#         losses.update({'commitment loss': commit_loss})

#         # Discriminator with detach
#         logits_x, features_stft_disc_x = self.msstftd_descrim(x.detach())
#         logits_G_x, features_stft_disc_G_x = self.msstftd_descrim(G_x.detach())
        
#         # Discrim Loss
#         loss_d = adversarial_d_loss(logits_x, logits_G_x)

#         loss_d.backward()
#         self.optimizer_d.step()

#         losses.update({'descriminator': loss_d})
#         return losses
        
#     def eval_step(self, batch):
#         self.train()
#         x, lengths_x = batch
#         device = next(self.parameters()).device

#         x = x.to(device)
#         lengths_x = lengths_x.to(device)

#         G_x, commit_loss = self.gen_step(x, lengths_x)
        
#         logits_x, features_stft_disc_x = self.msstftd_descrim(x)
#         logits_G_x, features_stft_disc_G_x = self.msstftd_descrim(G_x)
        
#         self.optimizer_g.zero_grad()
#         losses = self.gen_loss_fn.get_losses(
#             x, 
#             G_x, 
#             logits_G_x, 
#             features_stft_disc_x, 
#             features_stft_disc_G_x, 
#         )
#         losses.update({'commitment loss': commit_loss})

#         loss_d = adversarial_d_loss(logits_x, logits_G_x)

#         losses.update({'descriminator': loss_d})
#         return losses