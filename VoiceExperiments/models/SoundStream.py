import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from vector_quantize_pytorch import ResidualVQ

from VoiceExperiments.models.base import get_optimizer
from VoiceExperiments.modules.SoundStream import *

class SoundStream(nn.Module):
    # TODO: Add FiLM, add quantizer dropout
    def __init__(self, model_config, optimizer_configs):
        super(SoundStream, self).__init__()

        params = model_config['params']

        self.sr = params['sr']
        
        C = params['C']
        D = params['D']
        n_q = params['n_q']
        codebook_size = params['codebook_size']

        W = params['W']
        H = params['H']

        LAMBDA_ADV = params['LAMBDA_ADV']
        LAMBDA_FEAT = params['LAMBDA_FEAT']
        LAMBDA_REC = params['LAMBDA_REC']

        # Generator
        self.encoder = Encoder(C=C, D=D)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D)

        # Descriminator
        self.wave_disc = WaveDiscriminator(num_D=D, downsampling_factor=2)
        self.stft_disc = STFTDiscriminator(C=C, F_bins=W//2)
        
        # Optimizers
        self.optimizer_g = get_optimizer(optimizer_configs["gen"])(
            list(self.encoder.parameters()) + list(self.quantizer.parameters()) + list(self.decoder.parameters()),
            **optimizer_configs["gen"]["kwargs"]
        )

        self.optimizer_d = get_optimizer(optimizer_configs["descrim"])(
            list(self.wave_disc.parameters()) + list(self.stft_disc.parameters()),
            **optimizer_configs["descrim"]["kwargs"]
        )

        # Define losses
        self.criterion_g = lambda x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft: LAMBDA_ADV*adversarial_g_loss(features_stft_disc_G_x, features_wave_disc_G_x, lengths_stft, lengths_wave) + LAMBDA_FEAT*feature_loss(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft) + LAMBDA_REC*spectral_reconstruction_loss(x, G_x, self.sr)
        self.criterion_d = adversarial_d_loss

    def forward(self, x):
        e = self.encoder(x)
        e = torch.swapaxes(e, 1, 2)
        quantized, _, _ = self.quantizer(e)
        quantized = torch.swapaxes(quantized, 1, 2)
        o = self.decoder(quantized)
        return o
  
    def train_step(self, batch):
        self.train()
        x, lengths_x = batch
        device = next(self.parameters()).device

        x = x.to(device)
        lengths_x = lengths_x.to(device)

        G_x = self.forward(x)

        s_x = torch.view_as_real(torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=True)).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.view_as_real(torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=True)).permute(0, 3, 1, 2)
        
        lengths_stft = self.stft_disc.features_lengths(lengths_s_x)
        lengths_wave = self.wave_disc.features_lengths(lengths_x)
    

        # TODO: shape hacking
        G_x = G_x[:, :, :x.shape[2]]
        s_G_x = s_G_x[:, :, :, :s_x.shape[3]]

        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        
        features_stft_disc_G_x = self.stft_disc(s_G_x)
        features_wave_disc_G_x = self.wave_disc(G_x)

        loss_g = self.criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)

        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        
        features_stft_disc_G_x_det = self.stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = self.wave_disc(G_x.detach())
        
        loss_d = self.criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
      
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        return {"Loss_G": loss_g, "Loss_D": loss_d, "G_x": G_x}
    
    def eval_step(self, batch):
        self.eval()

        x, lengths_x = batch
        device = next(self.parameters()).device

        x = x.to(device)
        lengths_x = lengths_x.to(device)
    
        G_x = self.forward(x)
        
        s_x = torch.stft(x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        lengths_s_x = 1 + torch.div(lengths_x, 256, rounding_mode="floor")
        s_G_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024, device=device), return_complex=False).permute(0, 3, 1, 2)
        G_x = G_x[:, :, :x.shape[2]]
        s_G_x = s_G_x[:, :, :, :s_x.shape[3]]
        lengths_stft = self.stft_disc.features_lengths(lengths_s_x)
        lengths_wave = self.wave_disc.features_lengths(lengths_x)
        
        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)

        features_stft_disc_G_x = self.stft_disc(s_G_x)
        features_wave_disc_G_x = self.wave_disc(G_x)
        
        loss_g = self.criterion_g(x, G_x, features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x, features_wave_disc_G_x, lengths_wave, lengths_stft)
        
        features_stft_disc_x = self.stft_disc(s_x)
        features_wave_disc_x = self.wave_disc(x)
        
        features_stft_disc_G_x_det = self.stft_disc(s_G_x.detach())
        features_wave_disc_G_x_det = self.wave_disc(G_x.detach())
        
        loss_d = self.criterion_d(features_stft_disc_x, features_wave_disc_x, features_stft_disc_G_x_det, features_wave_disc_G_x_det, lengths_stft, lengths_wave)
    
        return {"Loss_G": loss_g, "Loss_D": loss_d, "G_x": G_x, "x": x}