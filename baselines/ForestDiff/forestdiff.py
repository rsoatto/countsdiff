import torch
import torch.nn as nn
import numpy as np
from ForestDiffusion import ForestDiffusionModel

class ForestDiffusionWrapper(nn.Module):
    def __init__(self, n_t=20, duplicate_K=100, repaint_iters=5, repaint_j=2, n_jobs = -1):
        super(ForestDiffusionWrapper, self).__init__()
        self.repaint_iters = repaint_iters
        self.repaint_j = repaint_j
        self.n_t = n_t
        self.duplicate_K = duplicate_K
        self.n_jobs = n_jobs
        
        
    def impute_data(self, counts, impute_mask, gene_names, valid_mask=None, labels=None):
        if isinstance(counts, torch.Tensor):
            counts = counts.numpy()
        if isinstance(impute_mask, torch.Tensor):
            impute_mask = impute_mask.numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        elif isinstance(labels, list):
            if isinstance(labels[0], torch.Tensor):
                labels = [cat_labels.numpy() for cat_labels in labels]
        
        num_genes = counts.shape[1]
        impute_and_invalid = (impute_mask == 1) | (valid_mask == 0) if valid_mask is not None else (impute_mask == 1)
        counts[impute_and_invalid] = np.nan
        labels = np.stack(labels, axis=1) if labels is not None else None
        Xy = np.concatenate([counts, labels], axis=1) if labels is not None else counts
        forest_model = ForestDiffusionModel(Xy, n_t=self.n_t, duplicate_K=self.duplicate_K, int_indices=list(range(num_genes)), cat_indices=list(range(num_genes, Xy.shape[1])), diffusion_type='vp', n_jobs = self.n_jobs) 
        
        imputed_full_data = forest_model.impute(repaint=True, r=self.repaint_iters, j=self.repaint_j, k=1)
        imputed_data =  imputed_full_data[:, :num_genes]
        imputed_data[~impute_mask.astype(bool)] = counts[~impute_mask.astype(bool)]
        return imputed_data
        