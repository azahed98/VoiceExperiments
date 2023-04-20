import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from encodec.balancer import Balancer

def adversarial_g_loss(logits):
    return F.relu(1-logits).mean()

def feature_loss(features_stft_disc_x, features_stft_disc_G_x,  lengths_stft):
    # paper divides feature difference for descrim k and layer l by mean( || D_k^l (x) ||_1 ), not included here but not sure why its needed?
    stft_loss = torch.stack([((feat_x-feat_G_x).abs().sum(dim=-1)/lengths_stft[i].view(-1,1,1)).sum(dim=-1).sum(dim=-1) for i, (feat_x, feat_G_x) in enumerate(zip(features_stft_disc_x, features_stft_disc_G_x))], dim=1)


    return stft_loss.mean()

def audio_reconstruction_loss(x, G_x):
    return (x -G_x).abs().sum()

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

# class DumbBalancer:
#     def __init__(self, weights):
#         self.weights = weights

#     def backward(self, losses, input):
#         pass

class EnCodecGenerativeLoss(nn.Module):
    def __init__(self, lambda_t, lambda_f, lambda_g, lambda_feat, lambda_w=0):
        self.loss_weights = {
            'wav_recontruction' : lambda_t, 
            'spectral_recontruction' : lambda_f, 
            'adversarial' : lambda_g,
            'feature' : lambda_feat,
            'vq_commitment' : lambda_w
        }
        
        self.balancer = Balancer(
            self.loss_weights
        ) 

    def backward(self, x, G_x, logits, features_stft_disc_x, features_stft_disc_G_x, sr, lengths_stft, commit_loss):
        losses = {
            'wav_recontruction' : audio_reconstruction_loss(x, G_x), 
            'spectral_recontruction' : spectral_reconstruction_loss(x, G_x, sr), 
            'adversarial' : adversarial_g_loss(logits),
            'feature' : feature_loss(features_stft_disc_x, features_stft_disc_G_x,  lengths_stft),
            'vq_commitment' : commit_loss
        }

        self.balancer.backward(losses, x)
        
def adversarial_d_loss(logits_x, logits_G_x, lengths_stft):
    real_stft_loss = F.relu(1-logits_x) 

    generated_stft_loss = F.relu(1+logits_G_x) 

    return real_stft_loss.mean() + generated_stft_loss.mean()

