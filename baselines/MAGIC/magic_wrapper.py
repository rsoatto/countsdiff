import magic
import scprep

import pandas as pd
import numpy as np
import torch

class MAGICWrapper:
    def __init__(self, 
                 knn = 40,
                 knn_max = None,
                 decay = 1,
                 t = 3, 
                 n_pca = 100,
                 solver = "exact", 
                 knn_dist = "euclidean",
                 n_jobs = 1,
                 random_state = None,
                 verbose = 1,
                 rescale = 10000):
        self.knn = knn
        self.knn_max = knn_max
        self.decay = decay
        self.t = t
        self.n_pca = n_pca
        self.solver = solver
        self.knn_dist = knn_dist
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.rescale = rescale
        self.magic_op = magic.MAGIC(knn = knn,
                                    knn_max = knn_max,
                                    decay = decay,
                                    t = t,
                                    n_pca = n_pca,
                                    solver = solver,
                                    knn_dist = knn_dist,
                                    n_jobs = n_jobs,
                                    random_state = random_state,
                                    verbose = verbose)

    def impute_data(self, counts, impute_mask, gene_names, valid_mask = None, labels = None):
        """
        Wrapper for MAGIC imputation
        """
        if isinstance(counts, torch.Tensor):
            counts = pd.DataFrame(counts.numpy())
        elif isinstance(counts, np.ndarray):
            counts = pd.DataFrame(counts)
        if isinstance(impute_mask, torch.Tensor):
            impute_mask = impute_mask.numpy()
        if isinstance(valid_mask, torch.Tensor):
            valid_mask = valid_mask.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        elif isinstance(labels, list):
            if isinstance(labels[0], torch.Tensor):
                labels = [cat_labels.numpy() for cat_labels in labels]

        counts.columns = gene_names
        original_sizes = scprep.measure.library_size(counts)
        rescale_factor = 1
        if isinstance(self.rescale, (int, float)):
            rescale_factor = self.rescale
        elif self.rescale == "median":
            rescale_factor = np.median(original_sizes)
        elif self.rescale == "mean":
            rescale_factor = np.mean(original_sizes)
        elif self.rescale == None:
            rescale_factor = 1

        # MAGIC should only see points that are later retained 
        norm_data = scprep.normalize.library_size_normalize(counts, rescale = self.rescale)
        norm_data = scprep.transform.sqrt(norm_data)

        # points to keep are valid and not imputed:
        retain_mask = valid_mask * (1 - impute_mask) if valid_mask is not None else (1 - impute_mask)

        data_magic = self.magic_op.fit_transform(norm_data*retain_mask, genes = "all_genes")

        imputed_targets = impute_mask * data_magic
        imputed_counts = round(imputed_targets/rescale_factor * original_sizes.values.reshape(-1,1))
        final_counts = imputed_counts + retain_mask * counts.values
        return final_counts
