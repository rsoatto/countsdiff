import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os

import sys
from pathlib import Path
import numpy as np
import torch
import h5py
import pandas as pd
import anndata
# Handle both direct execution and module execution     

from countsdiff.training.trainer import CountsdiffTrainer
from countsdiff.generation.generator import CountsdiffGenerator, CountsdiffImputer
from countsdiff.config.config import Config
from countsdiff.utils.metrics import scFID


import os, torch
import sys; sys.path.append(os.getcwd())
from baselines.ReMDM.remdm import ReMDM
from countsdiff.data.process_scrna import SingleCellDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, default='fetus', choices=['fetus','heart'])
args = parser.parse_args()
data_type = args.data_type

# Point this to your processed .h5 file used elsewhere in the notebook
if data_type == 'fetus':
    H5_PATH = "data/dnadiff/filtered_hca_data.hdf5"  
    train_ds = SingleCellDataset(H5_PATH, split='train', condition_keys=['cell_type','disease','sex','batch','development_day'])
    val_ds   = SingleCellDataset(H5_PATH, split='val',   condition_keys=['cell_type','disease','sex','batch','development_day'])
elif data_type == 'heart':
    H5_PATH = "data/dnadiff/filtered_heart_data.hdf5"
    train_ds = SingleCellDataset(H5_PATH, split='train', condition_keys=['batch', 'cell_type', 'gender', 'age'])
    val_ds   = SingleCellDataset(H5_PATH, split='val',   condition_keys=['batch', 'cell_type', 'gender', 'age'])

num_genes     = train_ds.counts.shape[1]
num_classes   = int(train_ds.counts.max().item()) + 1  # classes 0..max inclusive
all_num_classes = [train_ds.get_num_classes(k) for k in train_ds.condition_keys]

# Instantiate model; load from checkpoint if available
ckpt = f'data/dnadiff/checkpoints/remdm/{data_type}/'
if os.path.exists(ckpt + 'latest.pth'):
    remdm = ReMDM.from_checkpoint(ckpt + 'latest.pth', device=device)
else:
    remdm = ReMDM(num_genes, num_classes, all_num_classes, device=device)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
# Train a few epochs; saves every 1000 steps and at epoch end
remdm.fit(train_loader,
epochs=2 if data_type=='fetus' else 15,
save_every=1000,
resume=False,
verbose=True,
log_every=100,
eval_every=5000,
scfid_train_dataset=train_ds,
scfid_val_dataset=val_ds,
scfid_feature_model_path='data/dnadiff/2024-02-12-scvi-homo-sapiens/scvi.model',
scfid_num_samples=5000,
scfid_n_steps=100,
scfid_guidance_scale=1.0,
keep_intermediate=True,
checkpoint_path=ckpt)