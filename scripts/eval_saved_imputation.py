import json
import numpy as np
import pandas as pd
import torch
import argparse
import os
from scipy.stats import spearmanr
from tqdm import tqdm

from countsdiff.data.process_scrna import SingleCellDataset
from countsdiff.utils.metrics import scFID, compute_resampled_eval

def get_args():
    parser = argparse.ArgumentParser(description = "imputation")
    parser.add_argument("--baseline_model", type = str, default = None,
                        help = "baseline model choice, e.g: magic, gain, etc",
                        choices=["magic", "scidpm", "remdm", "remdm_full", "countsdiff", "mean"])
    
    parser.add_argument("--data_type", type = str, 
                        default = "fetus",
                        help = "either fetus or heart")
    
    parser.add_argument("--save_dir", type = str, 
                        default = "data/dnadiff/imputed_data", 
                        help = "directory to store imputed data")
    
    parser.add_argument("--mask_type", type = str, 
                        default = "MCAR")
    parser.add_argument("--dropout", type = float, default = 0.5)
    
    parser.add_argument("--data_path", type = str, default = None)
    
    parser.add_argument("--n_resamples", type = int, default = 10,
                        help = "number of resamples to compute metrics")

    parser.add_argument("--zero-shot", action="store_true", help = "whether to use zero-shot setting")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    assert args.baseline_model in {"magic", "gain", "misgan", "forestdiff", "scidpm", "countsdiff", "mean", "conditional_mean", "remdm", "remdm_full"}

    assert args.data_type in {"fetus", "heart", "zero_shot"}

    if args.data_type == "fetus":
        args.data_file = f'data/dnadiff/filtered_hca_data.hdf5'
        args.cond_keys = ["disease", "Development_day", "Batch", "cell_type", "sex"]
    if args.data_type == "heart":
        args.data_file = f'data/dnadiff/filtered_heart_data.hdf5'
        args.cond_keys = ["batch", "cell_type", "gender", "age"]
    if args.mask_type == "MNAR_low":
        args.mask_file = f'data/dnadiff/random_masks/{args.mask_type}_masks/{args.data_type}_dropout_{args.dropout}_low.npy'
    else:
        args.mask_file = f'data/dnadiff/random_masks/{args.mask_type}_masks/{args.data_type}_dropout_{args.dropout}.npy'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok = True)

    train_data = SingleCellDataset(args.data_file, split = "train", condition_keys = args.cond_keys)
    gene_names = train_data.gene_names

    print("Initializing metric...")
    obs_dict = train_data.get_obs_dict(unique = True)
    scfid = scFID(gene_names= gene_names, feature_model_path= "data/dnadiff/2024-02-12-scvi-homo-sapiens/scvi.model", categorical_covariates=obs_dict)
    
    if args.data_path is None:
        base_path = os.path.join(args.save_dir, f"{args.baseline_model}_{args.data_type}_results_{args.mask_type}_dropout_{args.dropout}.npy")
        if args.zero_shot:
            zs_path = os.path.join(args.save_dir, f"{args.baseline_model}_{args.data_type}_results_{args.mask_type}_dropout_{args.dropout}_zero_shot.npy")
            save_path = zs_path if os.path.exists(zs_path) else base_path
        else:
            save_path = base_path
    else:
        save_path = args.data_path

    imputed_data_multi = np.load(save_path)
    if imputed_data_multi.ndim == 2: 
        # single imputation case
        print("Single imputation detected.")
        imputed_data_multi = imputed_data_multi[..., np.newaxis]
    
    num_imputations = imputed_data_multi.shape[-1]
    print(f"evaluating {num_imputations} imputations:")
    test_data = SingleCellDataset(args.data_file, split = "test", condition_keys = args.cond_keys)
    impute_mask = np.load(args.mask_file).astype(bool)
    target_mask = impute_mask * ~(test_data.missingness_mask.numpy())
    
    if args.data_type == "fetus":
        n_samples = 50000
    elif args.data_type == "heart":
        n_samples = 20000
    print(f"Aggregating {num_imputations} imputations...")
    aggregated_results = np.mean(imputed_data_multi, axis = -1)
    results = compute_resampled_eval(scfid, aggregated_results, test_data.counts, target_mask, test_data.get_obs_dict(), n_samples, n_resamples = args.n_resamples)
    scfid.reset()
    json.dump(results, open(f"{args.save_dir}/results_{args.baseline_model}_{args.data_type}_{args.mask_type}_{args.dropout}_{num_imputations}imputations.json", "w"))
    print(f"Results saved to {args.save_dir}/results_{args.baseline_model}_{args.data_type}_{args.mask_type}_{args.dropout}_{num_imputations}imputations.json")
if __name__ == "__main__":
    main()
