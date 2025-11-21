import torch
import numpy as np
import argparse
import yaml
from torch.utils.data import DataLoader

from baselines.scIDPMs.main_model_table import scIDPMs
from baselines.scIDPMs.utils_table import genera
from baselines.scIDPMs.data_utils import SingleCellDatasetBaselines


class scIDPMWrapper():
    def __init__(self, 
                 data_dir,
                 max_arr,
                 config_path = "default.yaml",
                 device = 'cpu',
                 ckpt_file = 'checkpoints/scIDPMs/heart/checkpoint_epoch_120.ckpt',
                 batch_size = 72,
                 missing_ratio = 0.1,
                 impute_mask = None,
                 n_sample = 1):
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.model = scIDPMs(config, device).to(device)
        self.max_arr = max_arr
        self.n_sample = n_sample
        checkpoint = torch.load(ckpt_file, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.test_set = SingleCellDatasetBaselines(data_dir, "test", None, missing_ratio = missing_ratio, imputation_mask = impute_mask)
        self.test_set.counts = ((self.test_set.counts - 0 + 1)/(max_arr - 0 + 1)) * self.test_set.observed_mask
        self.test_loader = DataLoader(self.test_set, batch_size = batch_size)

    def impute_data(self):
        imputed_data = genera(self.model,
            self.test_loader,
            nsample = self.n_sample, 
            max_arr= self.max_arr, 
            gene_names = self.test_set.gene_names)
        
        return imputed_data, self.test_set


