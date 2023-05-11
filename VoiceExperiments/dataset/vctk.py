import torch
from torch import nn
from torchaudio.datasets import VCTK_092

import random

def collate_fn(batch):
    audios = [i[0].T for i in batch]
    # srs = [i[1] for i in batch]
    lengths = torch.tensor([elem.shape[0] for elem in audios])
    
    # audios shape after padding: (batch, 1, L) the 1 is for num channels
    return nn.utils.rnn.pad_sequence(audios, batch_first=True).permute(0, 2, 1), lengths

class VCTKDataset(VCTK_092):
    def __init__(self, audio_len=None, *args, **kwargs):
        super(VCTKDataset, self).__init__(*args, **kwargs)
        self.audio_len = None #samples
        self.collate_fn = collate_fn
        
    def _load_audio(self, file_path):
        audio = super(VCTKDataset, self)._load_audio(file_path)
        if self.audio_len and len(audio) > self.audio_len:
            start = random.randrange(len(audio) - self.audio_len)
            audio = audio[start:start + self.audio_len]
        return audio
    # @staticmethod
    # def collate_fn(batch):
    #     # TODO: Update collate to make universal
    #     audios = [i[0].T for i in batch]
    #     # srs = [i[1] for i in batch]
    #     lengths = torch.tensor([elem.shape[-1] for elem in audios])
    #     return nn.utils.rnn.pad_sequence(audios, batch_first=True)[:,:, 0][:, None, :], lengths