import torch
from torchaudio.datasets import VCTK_092


class VCTKDataset(VCTK_092):
    @staticmethod
    def collate_fn(batch):
        # TODO: Update collate to make universal
        audios = [i[0].T for i in batch]
        # srs = [i[1] for i in batch]
        lengths = torch.tensor([elem.shape[-1] for elem in audios])
        return nn.utils.rnn.pad_sequence(audios, batch_first=True)[:,:, 0][:, None, :], lengths