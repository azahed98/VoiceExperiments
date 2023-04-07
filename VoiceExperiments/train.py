import torch
import logging
import yaml
from yaml import Loader
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from VoiceExperiments.models import get_model#, get_optimizer
from VoiceExperiments.dataset import get_datasets
from VoiceExperiments.utils.logging import TensorBoardLogger

from torch.utils.tensorboard import SummaryWriter

# TODO: Find appropriate place to integrate collate_fn
#       may require writing custom replacement for ConcatDataset

def collate_fn(batch):
    audios = [i[0].T for i in batch]
    # srs = [i[1] for i in batch]
    lengths = torch.tensor([elem.shape[0] for elem in audios])
    return nn.utils.rnn.pad_sequence(audios, batch_first=True)[:,:, 0][:, None, :], lengths

def main(args):
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.model_config, 'r') as stream:
        model_cfg = yaml.load(stream, Loader)
    with open(args.training_config, 'r') as stream:
        training_cfg = yaml.load(stream, Loader)
    train_dataset, valid_dataset = get_datasets(training_cfg["datasets"])

    logging.info("Loaded datasets")
    
    opt_cfgs = training_cfg["optimizers"]
    model = get_model(model_cfg, opt_cfgs)
    model.to(device)
    BATCH_SIZE = training_cfg["BATCH_SIZE"]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=args.workers)
    
    logger = TensorBoardLogger(training_cfg['logging'], model_cfg=model_cfg, training_cfg=training_cfg)

    epoch = 0
    step = 0
    while training_cfg["max_epochs"] <= 0 or training_cfg["max_epochs"] > epoch:
        # Train
        model.train()
        
        for batch in tqdm(train_loader):
            results = model.train_step(batch)
            logger.log_train(results, epoch, step, model)
            step += 1
        
        # Val
        model.eval()
        
        for batch in tqdm(valid_loader):
            results = model.eval_step(batch)
            logger.log_val(results, epoch, step, model)
            step += 1

        epoch += 1

    logging.info("Done training")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--training_config', type=str)
    parser.add_argument('--workers', type=int, default=2)

    args = parser.parse_args()
    main(args)