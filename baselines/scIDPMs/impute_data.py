import torch
import numpy as np
import os
import argparse
import yaml
from torch.utils.data import DataLoader

from baselines.scIDPMs.main_model_table import scIDPMs
from baselines.scIDPMs.data_utils import SingleCellDatasetBaselines
from baselines.scIDPMs.utils_table import genera





parser = argparse.ArgumentParser(description="scIDM")
parser.add_argument("--config", type=str, default="baselines/scIDPMs/default.yaml")
parser.add_argument("--device", default="cpu", help="Device")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--missing_ratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--file_path", type=str, default="")
parser.add_argument("--label_path", type=str, default=None)
parser.add_argument('--att', type=str, default='MHA')
parser.add_argument('--n_genes', type=int, default=0)
parser.add_argument('--gpu_id', type = int, default = 0)
parser.add_argument('--data_dir', type = str, default = "data/dnadiff/filtered_hca_data.hdf5")

args = parser.parse_args()

config_path = args.config
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

if torch.cuda.is_available():
  dev = f"cuda:{args.gpu_id}"
else:
  dev = "cpu"
device = torch.device(dev)
args.device = device

model = scIDPMs(config, args.device).to(args.device)
checkpoint = torch.load("baselines/scIDPMs/checkpoints/scIDPMs/heart/checkpoint_epoch_120.ckpt", map_location=args.device)
model.load_state_dict(checkpoint['model_state_dict'])
test_set = SingleCellDatasetBaselines(args.data_dir, "test", None, missing_ratio = None, imputation_mask = "data/dnadiff/random_masks/MCAR_masks/heart_dropout_0.5.npy")
train_data = SingleCellDatasetBaselines(args.data_dir, split = 'train', missing_ratio = args.missing_ratio)
max_arr = np.max(train_data.counts.numpy(), axis = 0)

test_set.counts = ((test_set.counts - 0 + 1)/(max_arr - 0 + 1)) * test_set.observed_mask

test_loader = DataLoader(test_set, batch_size =72)
vals = genera(model,
       test_loader,
       nsample = 1, 
       max_arr= max_arr, 
       gene_names = train_data.gene_names)
np.save("baselines/scIDPMs/imputed_data/scid_heart_0.5.npy", vals)

