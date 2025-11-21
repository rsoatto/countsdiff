import argparse
import os
import numpy as np
import pandas as pd
import torch

from countsdiff.data.process_scrna import SingleCellDataset
from baselines.ForestDiff.forestdiff import ForestDiffusionWrapper
from countsdiff.utils.metrics import scFID

def parse_arguments():
    parser = argparse.ArgumentParser(description= "forestdiff")
    parser.add_argument('--repaint_iters', type = int, default = 1)
    parser.add_argument('--repaint_j', type = int, default = 1)
    parser.add_argument('--n_t', type= int, default = 20)
    parser.add_argument('--missing_mask', type = str, default = 'data/dnadiff/random_masks/MCAR_masks/heart_dropout_0.5.npy')
    parser.add_argument('--missing_rate', type = float, default = 0.5)
    parser.add_argument('--data_type', type = str, default = "heart")
    parser.add_argument('--n_samples', type = int, default = None)
    parser.add_argument('--duplicate_K', type = int, default = 1)
    parser.add_argument('--n_jobs', type = int, default = 40)
    args=parser.parse_args()
    return args

def evaluate_imputer(imputer, test_dataset, covariate_df_builder, impute_mask=None, num_samples=None, dropout_ratio=0.5):
    feature_model_path = "data/dnadiff/2024-02-12-scvi-homo-sapiens/scvi.model"
    scfid = scFID(gene_names = test_dataset.gene_names, categorical_covariates = test_dataset.get_obs_dict(unique = True), feature_model_path = feature_model_path)
    
    if num_samples is None:
        num_samples = len(test_dataset)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    first_batch = next(iter(dataloader))
    labels = first_batch[1]
    # labels = [label.cuda()[:num_samples] for label in labels]
    labels = [label.cpu()[:num_samples] for label in labels]
    missing_masks = first_batch[2]
    # missing_masks = missing_masks.cuda()[:num_samples]
    missing_masks = missing_masks[:num_samples]
    valid_mask = ~missing_masks
    ground_truth = first_batch[0]
    # ground_truth = ground_truth.cuda()[:num_samples]
    ground_truth = ground_truth.cpu()[:num_samples]

    
    if impute_mask is None:
        impute_mask = (torch.rand_like(ground_truth) < dropout_ratio)
    impute_mask = impute_mask * valid_mask
    masked = ground_truth * (~impute_mask).float()
    with torch.no_grad():
        print(f'Imputing {impute_mask.sum().item()} values out of {valid_mask.sum().item()} valid values ({100 * impute_mask.sum().item() / valid_mask.sum().item():.2f}%)')
        imputed_data = imputer.impute_data(counts=masked, gene_names = test_dataset.gene_names, labels=labels, valid_mask=valid_mask, impute_mask=impute_mask)
    
    if isinstance(imputed_data, np.ndarray):
        imputed_data = torch.from_numpy(imputed_data).to(device=ground_truth.device, dtype=ground_truth.dtype)
    
    torch.save(imputed_data, 'forestimputev2.pt')
    
    imputed_data[~impute_mask] = ground_truth[~impute_mask]
    

    imputed_vals = torch.masked_select(imputed_data, impute_mask.bool())
    actual_vals = torch.masked_select(ground_truth, impute_mask.bool())
    print(f'Computing metrics on {imputed_vals.shape[0]} imputed values')
    #Compute RMSE
    rmse = torch.sqrt(torch.mean((imputed_vals - actual_vals) ** 2)).item()
    print(f'RMSE: {rmse}')

    # raw bias
    raw_bias = torch.mean(imputed_vals - actual_vals).item()
    print(f'raw bias: {raw_bias}')

    # MAE
    mae = torch.mean(torch.abs(imputed_vals-actual_vals)).item()
    print(f'mae: {mae}')

    #Compute R2
    ss_res = torch.sum((actual_vals - imputed_vals) ** 2)
    ss_tot = torch.sum((actual_vals - torch.mean(actual_vals)) ** 2)
    r2 = (1 - ss_res / ss_tot).item()
    print(f'R2: {r2}')
    
    #Compute scFID
    scfid.update(ground_truth.cpu().numpy(), test_dataset.get_obs_dict(), True)
    scfid.update(imputed_data.cpu().numpy(), test_dataset.get_obs_dict(), False)
    scfid_value = scfid.compute()
    print(f'scFID: {scfid_value}')
    
    return {
        'imputed_data': imputed_data,
        'ground_truth': ground_truth,
        'impute_mask': impute_mask,
        'rmse': rmse,
        'mae': mae,
        'raw_bias': raw_bias,
        'r2': r2,
    
    }
    

def main():
    args = parse_arguments()
    if args.data_type == "fetus":
        data_path = f"data/dnadiff/filtered_hca_data.hdf5"
        cond_keys = ["cell_type",  "batch", "development_day", 'sex', "disease"]
    elif args.data_type == "heart":
        data_path = f"data/dnadiff/filtered_heart_data.hdf5"
        cond_keys = ["cell_type",  "batch", "age", 'gender']
    
    test_dataset = SingleCellDataset(data_path, "test", condition_keys= cond_keys)
    forrestdiff_imputer = ForestDiffusionWrapper(repaint_iters=args.repaint_iters, repaint_j=args.repaint_j, n_t=args.n_t, duplicate_K = args.duplicate_K, n_jobs= args.n_jobs)
    if args.missing_mask:
        impute_mask = torch.from_numpy(np.load(args.missing_mask).astype(bool))
    else:
        impute_mask = None
    results = evaluate_imputer(forrestdiff_imputer, test_dataset, test_dataset.get_conditions_df(), impute_mask = impute_mask)
    return results

if __name__ == "__main__":
    res = main()
    print(res)
