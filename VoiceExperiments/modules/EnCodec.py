import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from encodec.balancer import Balancer

def adversarial_g_loss(logits):
    adv_loss = 0
    for logit in logits:
        adv_loss += F.relu(1-logit).sum(dim=3).squeeze().mean() # summing across frequency domain. Ideally use fully connected layer instead
    return adv_loss

def feature_loss(features_multi_stft_disc_x, features_multi_stft_disc_G_x):
    # paper divides feature difference for descrim k and layer l by mean( || D_k^l (x) ||_1 ), not included here but not sure why its needed?
    # stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1)
    
    # || D_k^l(x) - D_k^l(xhat) ||_1
    stft_loss = 0
    for features_stft_disc_x, features_stft_disc_G_x in zip(features_multi_stft_disc_x, features_multi_stft_disc_G_x):
        stft_loss += torch.stack([
            (feat_x - feat_G_x).abs().sum(dim=-1) / feat_x.abs().sum(dim=-1).mean()
            for i, (feat_x, feat_G_x) 
            in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))
        ]).sum()

    stft_loss /= (len(features_multi_stft_disc_G_x) * len(features_multi_stft_disc_G_x[0]))
    return stft_loss

def audio_reconstruction_loss(x, G_x):
    return (x - G_x).abs().sum()

def spectral_reconstruction_loss(x, G_x, sr, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    L = 0
    for i in range(5,12):
        s = 2**i
        alpha_s = (s/2)**0.5
        melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=8, wkwargs={"device": device}).to(device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        
        l1_loss = (S_x-S_G_x).abs().sum()
        l2_loss = (((S_x-S_G_x)**2).sum()**0.5)
        L += l1_loss + alpha_s * l2_loss

    return L


class EnCodecGenerativeLoss:
    def __init__(self, sample_rate, lambda_t=0.1, lambda_f=1, lambda_g=3, lambda_feat=3, lambda_w=1,):
        # Losses
        # - Reconstruction Loss (time and frequency)
        # - Discriminative Loss
        # - VQ Commitment loss
        # - Balancer (not a loss, but a way of combining)
        # Also need to consider multi-bandwidth training
        #   note from paper: 
        #       (for Multi STFT Descrim) descriminator tend to overpower easily the decoder
        #       we update its weight with a probability of 2/3 at 24 kHz and 0.5 at 48 kHz
        self.loss_weights = {
            'wav_recontruction' : lambda_t, 
            'spectral_reconstruction' : lambda_f, 
            'adversarial' : lambda_g,
            'feature' : lambda_feat,
        }

        self.lambda_w = lambda_w # commitment loss not part of balancer

        self.balancer = Balancer(
            self.loss_weights
        ) 

        self.sample_rate = sample_rate

    def backward(self, x, G_x, logits, features_stft_disc_x, features_stft_disc_G_x, commit_loss, params, use_balancer=True):
        losses = self.get_losses(x, G_x, logits, features_stft_disc_x, features_stft_disc_G_x)

        if use_balancer:
            for param in params:
                self.balancer.backward(losses, param)
        else:
            loss = sum([self.loss_weights[k] * losses[k] for k in losses])
            loss.backward()
        
        commit_loss = self.lambda_w * commit_loss
        
        commit_loss.backward()

        return losses
    
    def get_losses(self, x, G_x, logits, features_stft_disc_x, features_stft_disc_G_x):
        losses = {
            'wav_recontruction' : audio_reconstruction_loss(x, G_x), 
            'spectral_reconstruction' : spectral_reconstruction_loss(x, G_x, self.sample_rate), 
            'adversarial' : adversarial_g_loss(logits),
            'feature' : feature_loss(features_stft_disc_x, features_stft_disc_G_x),
        }
        return losses
    
def adversarial_d_loss(logits_x, logits_G_x):
    for logit in logits_x:
        real_stft_loss = F.relu(1-logit).sum(dim=3).squeeze().mean()

    for logit in logits_G_x:
        generated_stft_loss = F.relu(1+logit).sum(dim=3).squeeze().mean()

    return real_stft_loss.mean() + generated_stft_loss.mean()

