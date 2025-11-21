import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import argparse

from data_utils import SingleCellDatasetBaselines
from utils_table import train
from main_model_table import scIDPMs

def main(args):
    """
    main function to run scIDPMs training for scRNA-seq data.
    """
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(os.getcwd())

    print(f"loading configs from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = args.missing_ratio
    train_data = SingleCellDatasetBaselines(args.data_dir, split = 'train', missing_ratio = args.missing_ratio)
    val_data = SingleCellDatasetBaselines(args.data_dir, split = 'val', missing_ratio = args.missing_ratio)

    print("Normalizing data...")
    max_arr = np.max(train_data.counts.numpy(), axis = 0)

    train_data.counts = ((train_data.counts - 0 + 1)/(max_arr - 0 + 1)) * train_data.observed_mask
    val_data.counts = ((val_data.counts - 0 + 1)/(max_arr - 0 + 1)) * val_data.observed_mask

    train_dataloader = DataLoader(train_data, batch_size = config["train"]["batch_size"], shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = config["train"]["batch_size"], shuffle = False)

    if torch.cuda.is_available():
        device = f'cuda:{args.gpu_id}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    print("Loading model...")
    model = scIDPMs(config, device).to(device)


    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p0 = int(0.25 * config["train"]["epochs"])
    p1 = int(0.5 * config["train"]["epochs"])
    p2 = int(0.75 * config["train"]["epochs"])
    p3 = int(0.9 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p0, p1, p2, p3], gamma=0.1
    )
    start_epoch = 0
    best_val_loss = 1e8

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_valid_loss"]
        
        print(f"Loaded checkpoint from epoch {start_epoch}.")
    else:
        print("Starting training from scratch.")
        

    print("Starting training...")
    train(model = model, 
          config = config["train"], 
          train_loader = train_dataloader, 
          optimizer = optimizer,
          lr_scheduler = lr_scheduler,
          best_val_loss = best_val_loss,
          start_epoch = start_epoch,
          valid_loader = val_dataloader, 
          foldername = args.checkpoint_dir, 
          seed = 42)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train scIDPMs Model.")

    parser.add_argument('--config_path', type=str, default = "default.yaml",
                        help='Path to the YAML configuration file.')
    parser.add_argument('--data_dir', type=str, default='filtered_heart_data.hdf5',
                        help='Path to HDF5 data directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/scIDPMs/heart',
                        help='Directory to save model checkpoints and logs.')
    parser.add_argument('--missing_ratio', type = float, default = 0.1,
                        help = 'Ratio of masked data for training imputation.')
    parser.add_argument('--gpu_id', type=int, default = 0)

    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to resume training from.')

    parsed_args = parser.parse_args()
    main(parsed_args)