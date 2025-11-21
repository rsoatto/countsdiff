import torch
import numpy as np
import pandas as pd
import os
import argparse
import sys

from typing import Dict, List, Any

project_root = "/your/project/root"

sys.path.append(project_root)

os.chdir(project_root) # You can keep this if you need it for file paths

from baselines.MAGIC.magic_wrapper import MAGICWrapper

from countsdiff.utils.metrics import scFID

from countsdiff.data.process_scrna import SingleCellDataset


def main(args):

    assert args.min_k < args.max_k
    best_k = args.min_k
    best_scfid = 1e8
    
    output_dir = args.save_dir
    data_file = "best_imputated_data.npy"
    mask_file = "mask.npy"
    print("Loading dataset....")
    train_data = SingleCellDataset(args.train_file, split = "train", condition_keys = args.condition_keys)
    test_data = SingleCellDataset(args.test_file, split = "test", condition_keys = args.condition_keys)
    gene_names = test_data.gene_names
    # num_elems = np.prod(data.counts.shape)
    # packed_array = np.load(args.mask_file)
    # impute_mask = np.unpackbits(packed_array)[:num_elems].reshape(data.counts.shape).astype(bool)

    impute_mask = np.random.rand(*test_data.counts.shape) < args.dropout_rate
    train_mask = np.ones(train_data.counts.shape, dtype = bool)
    train_size = len(train_data)
    valid_mask = ~test_data.missingness_mask.numpy()
    target_mask = impute_mask * valid_mask
    total_counts = np.concatenate((train_data.counts, test_data.counts), axis = 0)
    total_impute_mask = np.concatenate((~train_mask, target_mask), axis = 0)
    total_valid_mask = np.concatenate((train_mask, valid_mask))
 

    print("Initializing metric...")
    obs_dict = train_data.get_obs_dict(unique = True)
    scfid = scFID(gene_names= gene_names, feature_model_path= "/your/project/root/src/countsdiff/2024-02-12-scvi-homo-sapiens/scvi.model", categorical_covariates=obs_dict)
    
    for knn in range(args.min_k, args.max_k, args.step_k):
        mwrapper = MAGICWrapper(knn = knn, n_jobs = args.n_jobs)

        # target mask should be intersection of imputed places and valid places
        
        imputed_data = mwrapper.impute_data(total_counts, total_impute_mask, gene_names, total_valid_mask)

        imputed_test_data = imputed_data.values[train_size:, :]
        

        imputed_vals = torch.masked_select(torch.from_numpy(imputed_test_data), torch.Tensor(target_mask).bool())
        actual_vals = torch.masked_select(test_data.counts, torch.Tensor(target_mask).bool())

        raw_bias = torch.mean(imputed_vals - actual_vals).item()
        mae = torch.mean(torch.abs(imputed_vals-actual_vals)).item()
        rmse = torch.sqrt(torch.mean((imputed_vals - actual_vals) ** 2)).item()

        ss_res = torch.sum((actual_vals - imputed_vals) ** 2)
        ss_tot = torch.sum((actual_vals - torch.mean(actual_vals)) ** 2)
        r2 = (1 - ss_res / ss_tot).item()


        scfid.update(test_data.counts.numpy(), test_data.get_obs_dict(unique = False), True)
        scfid.update(imputed_test_data, test_data.get_obs_dict(unique = False), False)
        score = scfid.compute().item()


        if score<best_scfid:
            best_scfid = score
            best_k = knn
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            np.save(os.path.join(output_dir, data_file), imputed_test_data)
            np.save(os.path.join(output_dir, mask_file), target_mask)

    return best_scfid, best_k, rmse, r2, raw_bias, mae







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval MAGIC imputation")

    parser.add_argument("--mask_file", type = str,
    default = "/your/project/root/src/countsdiff/data/random_masks/0.4_masked.npy",
    help = "path to packed np array of imputation mask")

    parser.add_argument("--train_file", type = str,
    default = "/your/project/root/src/countsdiff/data/filtered_hca_data.hdf5",
    help = "path to hdf5 data file")

    parser.add_argument("--test_file", type = str, 
                        default = "/your/project/root/src/countsdiff/data/filtered_hca_data.hdf5", 
                        help = "path to hdf5 data file for evaluation")

    parser.add_argument("--min_k", type = int, default = 15, help = "minimum k for knn")

    parser.add_argument("--max_k", type = int, default = 60, help = "maximum k for knn for MAGIC")

    parser.add_argument("--step_k", type = int, default = 3, help = "step size for sweep for k")

    parser.add_argument("--condition_keys", type = list[str], default = ["cell_type", "Batch", "Development_day", "disease", "sex"], help = "list of conditional covariates")

    parser.add_argument("--n_jobs", type = int, default = 1, help = "number of jobs to run MAGIC")

    parser.add_argument("--dropout_rate", type = float, default = 0.4, help = "amount of dropout")

    parser.add_argument("--save_dir", type = str, default = "/your/project/root/baselines/MAGIC/imputed_data")
    args = parser.parse_args()

    print(main(args))