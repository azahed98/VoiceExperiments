import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from vector_quantize_pytorch import ResidualVQ

from VoiceExperiments.modules.SoundStream import *

class SoundStreamGenerator(nn.Module):
    def __init__(self, C, D, n_q, codebook_size):
        self.encoder = Encoder(C=C, D=D)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D)

    def forward(self, x):
        e = self.encoder(x) # (B, S, D)
        e = torch.swapaxes(e, 1, 2) # (B, D, S)
        quantized, _, _ = self.quantizer(e) # (B, D, S)
        quantized = torch.swapaxes(quantized, 1, 2) # (B, S, D)
        o = self.decoder(quantized) # (B, 1, T)
        return o
    

class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor**i}": self.model[f"disc_{self.downsampling_factor**i}"].features_lengths(torch.div(lengths, 2**i, rounding_mode="floor")) for i in range(self.num_D)
        }
    
    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results[f"disc_{self.downsampling_factor**i}"] = disc(x)
            x = self.downsampler(x)
        return results


class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32,  N=C,   m=2, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*C, N=2*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C, N=8*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C,  N=8*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*C, out_channels=1,
                      kernel_size=(F_bins//2**6, 1))
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths-6,
            lengths-6,
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map

