import torch
import logging
import yaml
import argparse
import torch.nn as nn

from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from yaml import Loader
from tqdm import tqdm

# from VoiceExperiments.models import get_model#, get_optimizer
from VoiceExperiments.pipelines import get_pipeline
from VoiceExperiments.dataset import get_dataset
from VoiceExperiments.utils.logging import TensorBoardLogger, tensor_dict_to_numpy, del_results

# from torch.utils.tensorboard import SummaryWriter

# TODO: Find appropriate place to integrate collate_fn
#       may require writing custom replacement for ConcatDataset

# def collate_fn(batch):
#     audios = [i[0].T for i in batch]
#     # srs = [i[1] for i in batch]
#     lengths = torch.tensor([elem.shape[0] for elem in audios])
    
#     # audios shape after padding: (batch, 1, L) the 1 is for num channels
#     return nn.utils.rnn.pad_sequence(audios, batch_first=True).permute(0, 2, 1), lengths

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.pipeline_config, 'r') as stream:
        pipeline_cfg = edict(yaml.load(stream, Loader))
    with open(args.training_config, 'r') as stream:
        training_cfg = edict(yaml.load(stream, Loader))
    train_dataset, valid_dataset, collate_fn = get_dataset(training_cfg.dataset)

    logging.info("Loaded dataset")
    
    opt_cfgs = training_cfg.optimizers
    pipeline = get_pipeline(pipeline_cfg, opt_cfgs)
    pipeline.to(device)
    if args.compile:
        pipeline.compile()

    BATCH_SIZE = training_cfg.BATCH_SIZE

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=args.workers)
    
    logger = TensorBoardLogger(training_cfg.logging, pipeline)
    clear_cache_every_n_steps = training_cfg.logging.clear_cache_every_n_steps

    epoch = 0
    step = 0
    while training_cfg.max_epochs <= 0 or training_cfg.max_epochs > epoch:
        # # Train
        pipeline.train()
        
        for batch in tqdm(train_loader):
            results = pipeline.train_step(batch)
            del_tensors = (clear_cache_every_n_steps > 0) and (step % clear_cache_every_n_steps == 0)
            results = tensor_dict_to_numpy(results, del_tensors=del_tensors)
            logger.log_train(results, epoch, step)
            step += 1
        del_results(results)
        # Val
        pipeline.eval()
        
        val_results = []
        val_step = 0
        for batch in tqdm(valid_loader):
            results = pipeline.eval_step(batch)
            del_tensors = (clear_cache_every_n_steps > 0) and (val_step % clear_cache_every_n_steps == 0)
            results = tensor_dict_to_numpy(results, del_tensors=del_tensors)
            val_results.append(results)

            val_step += 1
        del_results(results)
        logger.log_val(val_results, epoch, step)
        
        epoch += 1

    logging.info("Done training")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_config', type=str)
    parser.add_argument('--training_config', type=str)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--compile', action="store_true")

    args = parser.parse_args()
    main(args)