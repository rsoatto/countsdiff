import json
import numpy as np
import pandas as pd
import torch
import argparse
import os
import warnings
from scipy.stats import spearmanr
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

from baselines.MAGIC.magic_wrapper import MAGICWrapper
from baselines.scIDPMs.scidpm_wrapper import scIDPMWrapper
from countsdiff.generation.generator import CountsdiffImputer
from baselines.ReMDM.remdm import ReMDM
from countdiff.data.process_scrna import SingleCellDataset



def get_args():
    parser = argparse.ArgumentParser(description = "imputation")
    parser.add_argument("--baseline_model", type = str, default = None,
                        help = "baseline model choice, e.g: magic, gain, etc",
                        choices=["magic", "scidpm", "remdm", "countsdiff", "mean", "conditional_mean"])
    
    parser.add_argument("--data_type", type = str, 
                        default = "fetus",
                        help = "either fetus or heart")
    
    parser.add_argument("--save_dir", type = str, 
                        default = "data/dnadiff/imputed_data", 
                        help = "directory to store imputed data")
    
    parser.add_argument("--mask_type", type = str, 
                        default = "MCAR")
    parser.add_argument("--dropout", type = float, default = 0.5)
    parser.add_argument("--gpu_id", type = int, default = 0)
    parser.add_argument("--num_imputations", type = int, default = 5, help = "number of multiple imputations to perform")
    
    

    # MAGIC only args:
    parser.add_argument("--magic_knn", type = int, default = 40, help = "k for knn for MAGIC")
    parser.add_argument("--magic_n_jobs", type = int, default = 1, help = "number of jobs to run MAGIC")

    # scidpm only args:
    parser.add_argument("--scidpm_trained", type = bool, default = True)
    parser.add_argument('--scidpm_checkpoint_dir', type = str, default = "baselines/scIDPMs/checkpoints/scIDPMs")
    parser.add_argument('--final_scidpm_checkpoint', type = int, default = 65)
    parser.add_argument('--scidpm_batch_size', type = int, default = 32)
    parser.add_argument('--scidpm_config_path', type = str, default = 'baselines/scIDPMs/default.yaml')
    
    # countsdiff only args:
    # defaults to optimal values for fetus
    parser.add_argument("--run-id", type = str, default = None, help = "neptune run id")
    parser.add_argument("--eta-rescale", type = float, default = 0.02)
    parser.add_argument("--n-steps", type = int, default = 20)
    parser.add_argument("--repaint-num-iters", type = int, default = 3)
    parser.add_argument("--repaint-jump", type = int, default = 20)
    parser.add_argument("--guidance-scale", type = float, default = 0.5)
    
    # zero-shot mode (as in all_imputation.py / eval_saved_imputation.py)
    parser.add_argument("--zero-shot", action="store_true", help = "whether to use zero-shot setting")
    
    # ReMDM only args
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to ReMDM checkpoint (.pth)")



    parser.add_argument("--parallel", action="store_true",
                        help="Run multiple imputations in parallel across GPUs.")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated list of GPU ids to use (e.g., '0,1,2'). "
                             "Defaults to all visible GPUs.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Max concurrent workers. Defaults to number of GPUs provided.")
    return parser.parse_args()




def impute_magic(args, train_data):
    
    train_size = len(train_data)
    test_data = SingleCellDataset(args.data_file, split = "test", condition_keys = args.cond_keys)
    impute_mask = np.load(args.mask_file).astype(bool) # True for points to impute
    train_mask = np.ones(train_data.counts.shape, dtype = bool) # True for train_set points
    valid_mask = ~test_data.missingness_mask.numpy()    # True for non-missing points
    target_mask = impute_mask * valid_mask      # imputation targets are imputed and valid
    total_counts = np.concatenate((train_data.counts, test_data.counts), axis = 0) # combining train and test data
    total_impute_mask = np.concatenate((~train_mask, target_mask), axis = 0) # only impute targets
    total_valid_mask = np.concatenate((train_mask, valid_mask)) # all valid points for later construction
    
    mwrapper = MAGICWrapper(knn = args.magic_knn, n_jobs = args.magic_n_jobs)

    imputed_data = mwrapper.impute_data(total_counts, total_impute_mask, test_data.gene_names, total_valid_mask)

    imputed_test_data = imputed_data.values[train_size:, :]

    return imputed_test_data

def impute_scidpm(args, max_arr):
    scwrapper = scIDPMWrapper(data_dir = args.data_file,
                              max_arr = max_arr,
                              config_path = args.scidpm_config_path,
                              device = args.device,
                              ckpt_file = args.scidpm_checkpoint_path,
                              batch_size = args.scidpm_batch_size,
                              impute_mask = args.mask_file)
    
    all_imputed_data, test_set = scwrapper.impute_data()
    impute_mask = np.load(args.mask_file).astype(bool)
    

    target_mask = impute_mask*test_set.observed_mask
    imputed_data = np.round(all_imputed_data) * target_mask + ~target_mask*test_set.counts.numpy()
    return imputed_data


def impute_countsdiff(args):
    run_id = args.run_id
    if run_id is None:
        raise ValueError("A valid run is needed")
    
    # In zero-shot, we cannot guide on mismatched conditions
    if getattr(args, "zero_shot", False):
        args.guidance_scale = 0.0
    
    countsdiff_imputer = CountsdiffImputer(
        run_id=args.run_id,
        device=args.device,
        n_steps=args.n_steps,
        guidance_scale=args.guidance_scale,
        repaint_num_iters=args.repaint_num_iters,
        repaint_jump=args.repaint_jump,
        sigma_method="rescaled",
        sigma_kwargs={"eta_rescale": args.eta_rescale}
    )
    
    if getattr(args, "zero_shot", False):
        test_data = SingleCellDataset(
            args.data_file,
            split="test",
            condition_keys=args.cond_keys
        )
    else:
        test_data = SingleCellDataset(
            args.data_file,
            split="test",
            condition_keys=countsdiff_imputer.generator.trainer.data_config.get('condition_keys', [])
        )
    
    impute_mask = np.load(args.mask_file).astype(bool)           # True for points to impute
    valid_mask  = ~test_data.missingness_mask.numpy()            # True for non-missing points
    target_mask = impute_mask & valid_mask                       # imputation targets are imputed and valid
    test_data.add_masks(target_mask)

    batch_size = 256
    dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    total_batches = max(1, len(dataloader))  # guard div-by-zero

    # dataloader-level progress (shared dict managed by parent)
    has_progress = hasattr(args, "progress_dict") and hasattr(args, "progress_idx")
    if has_progress:
        try:
            args.progress_dict[args.progress_idx] = 0.0
        except Exception:
            has_progress = False

    imputed_data = None
    for batch_idx, batch in enumerate(dataloader, start=1):
        batch_counts, batch_labels, batch_missingness, impute_mask = batch
        batch_counts = batch_counts.to(args.device)
        batch_valid  = ~batch_missingness.to(args.device)
        impute_mask  = impute_mask.to(args.device)
        batch_labels = [x.to(args.device) for x in batch_labels]
        if getattr(args, "zero_shot", False):
            # Supply zero labels for each condition expected by the model
            model_conds = countsdiff_imputer.generator.trainer.data_config.get('condition_keys', [])
            batch_labels = [torch.zeros_like(batch_labels[0]) for _ in model_conds]

        imputed_batch = countsdiff_imputer.impute_data(
            batch_counts, batch_valid, impute_mask, batch_labels
        )
        if imputed_data is None:
            imputed_data = imputed_batch
        else:
            imputed_data = np.concatenate((imputed_data, imputed_batch), axis=0)

        if has_progress:
            args.progress_dict[args.progress_idx] = batch_idx / total_batches

    if isinstance(imputed_data, torch.Tensor):
        imputed_data = imputed_data.numpy()



    if has_progress:
        args.progress_dict[args.progress_idx] = 1.0

    return imputed_data  


def impute_remdm(args):
    if not args.checkpoint_path or not os.path.exists(args.checkpoint_path):
        raise ValueError(f"ReMDM checkpoint_path not found or not provided: {args.checkpoint_path}")

    imputer = ReMDM.from_checkpoint(args.checkpoint_path, device=args.device)
    # In zero-shot, we cannot guide on mismatched conditions
    gs = 0.0 if getattr(args, "zero_shot", False) else float(args.guidance_scale)
    imputer.update_hyperparameters(
        n_steps=int(args.n_steps),
        device=str(args.device),
        guidance_scale=gs,
        sigma_method='rescaled',
        sigma_kwargs={'eta_rescale': float(args.eta_rescale)},
        repaint_num_iters=int(args.repaint_num_iters),
        repaint_jump=int(args.repaint_jump),
        batch_size=256,
    )

    test_data = SingleCellDataset(args.data_file, split="test", condition_keys=args.cond_keys)
    impute_mask = np.load(args.mask_file).astype(bool)
    valid_mask  = ~test_data.missingness_mask.numpy()
    target_mask = impute_mask & valid_mask
    test_data.add_masks(target_mask)

    dataloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)
    total_batches = max(1, len(dataloader))

    has_progress = hasattr(args, "progress_dict") and hasattr(args, "progress_idx")
    if has_progress:
        try:
            args.progress_dict[args.progress_idx] = 0.0
        except Exception:
            has_progress = False

    out_parts = []
    for batch_idx, batch in enumerate(dataloader, start=1):
        batch_counts, batch_labels, batch_missingness, batch_impute_mask = batch
        batch_counts = batch_counts.to(args.device)
        batch_valid  = (~batch_missingness).to(args.device).bool()
        batch_impute_mask = batch_impute_mask.to(args.device).bool()
        batch_masked = batch_counts.clone()
        batch_masked[batch_impute_mask] = 0.0
        batch_labels = [x.to(args.device) for x in batch_labels]
        if getattr(args, "zero_shot", False):
            # Create zero labels for each embedder expected by the model
            n_conds = len(imputer.model.embedders)
            proto = batch_labels[0] if isinstance(batch_labels, list) and len(batch_labels) > 0 else torch.zeros(batch_counts.shape[0], dtype=torch.long, device=args.device)
            zero_like = torch.zeros_like(proto, dtype=torch.long, device=args.device)
            batch_labels = [zero_like for _ in range(n_conds)]

        with torch.no_grad():
            imputed_np = imputer.impute_data(
                counts=batch_masked,
                valid_mask=batch_valid,
                impute_mask=batch_impute_mask,
                labels=batch_labels,
            )

        imputed = torch.from_numpy(imputed_np) if isinstance(imputed_np, np.ndarray) else imputed_np
        imputed = imputed.to(batch_counts.device)
        imputed[~batch_impute_mask] = batch_counts[~batch_impute_mask]
        out_parts.append(imputed.detach().cpu().numpy())

        if has_progress:
            args.progress_dict[args.progress_idx] = batch_idx / total_batches

    if has_progress:
        args.progress_dict[args.progress_idx] = 1.0

    return np.concatenate(out_parts, axis=0)

def impute_mean(args, train_data):
    test_data = SingleCellDataset(args.data_file, split = "test", condition_keys = args.cond_keys)
    impute_mask = np.load(args.mask_file).astype(bool) # True for points to impute
    valid_mask = ~test_data.missingness_mask.numpy()    # True for non-missing points
    target_mask = impute_mask & valid_mask
    #compute means of valid genes
    gene_sums = test_data.counts.sum(axis = 0).numpy()
    valid_sums = valid_mask.sum(axis = 0)
    gene_means = gene_sums/valid_sums

    imputed_data = test_data.counts.numpy().copy()
    imputed_data[target_mask] = np.take(gene_means, np.where(target_mask)[1])

    return imputed_data

def impute_conditional_means(args, train_data):
    """
    Impute test data by conditional means:
      - Build a lookup from TRAIN: (covariate combo) -> per-gene mean, using only valid entries.
      - For each TEST row: use its covariate combo's per-gene means where available;
        otherwise fall back to GLOBAL per-gene mean (from TRAIN).
    """
    # ---- TRAIN STATS (lookup) ----
    # Shapes
    train_counts = train_data.counts.numpy()               # [N_train, G]
    train_valid  = (~train_data.missingness_mask.numpy())  # [N_train, G]  (True = valid / observed)
    N_train, G = train_counts.shape

    # Per-gene GLOBAL means from TRAIN (valid-only)
    train_valid_sums   = (train_counts * train_valid).sum(axis=0)       # [G]
    train_valid_counts = train_valid.sum(axis=0)                         # [G]
    global_gene_means  = np.divide(
        train_valid_sums,
        np.clip(train_valid_counts, 1, None),
        where=train_valid_counts > 0
    )
    # (If some gene never appears valid in train, set to 0.0)
    global_gene_means[np.isnan(global_gene_means)] = 0.0

    # Build mapping from covariate combo -> row indices in TRAIN
    train_obs = train_data.get_obs_dict()  # per-row covariates
    cov_df = pd.DataFrame({k: np.asarray(v) for k, v in train_obs.items() if k in args.cond_keys})
    cov_df["__row__"] = np.arange(N_train)
    grouped = cov_df.groupby(args.cond_keys, dropna=False)

    # Compute per-group per-gene means & counts (valid-only)
    group_means = {}   # key(tuple) -> np.ndarray[G]
    group_counts = {}  # key(tuple) -> np.ndarray[G] (#valid contributions per gene)
    for key, idx in grouped.indices.items():
        idx = np.asarray(idx, dtype=int)
        sub_counts = train_counts[idx]               # [n_k, G]
        sub_valid  = train_valid[idx]                # [n_k, G]
        sums   = (sub_counts * sub_valid).sum(axis=0)
        counts = sub_valid.sum(axis=0)
        means  = np.divide(
            sums,
            np.clip(counts, 1, None),
            where=counts > 0
        )
        means[np.isnan(means)] = 0.0
        # Normalize key to a tuple (pandas may give scalar for single key)
        if not isinstance(key, tuple):
            key = (key,)
        group_means[key]  = means
        group_counts[key] = counts

    # ---- TEST IMPUTATION ----
    test_data   = SingleCellDataset(args.data_file, split="test", condition_keys=args.cond_keys)
    impute_mask = np.load(args.mask_file).astype(bool)           # True = impute target
    valid_mask  = (~test_data.missingness_mask.numpy())          # True = non-missing in data
    target_mask = impute_mask & valid_mask                       # only impute targets that are actually missing

    imputed_data = test_data.counts.numpy().copy()                # we will write into this copy
    N_test = imputed_data.shape[0]
    test_obs  = test_data.get_obs_dict()                         # per-row covariates for TEST

    # Create per-row keys for TEST
    def row_key(i):
        # Snapshot the row's covariate combo in same ordering as args.cond_keys
        tup = tuple(test_obs[k][i] for k in args.cond_keys)
        # Ensure it's hashable & consistent with train keys (strings/numbers are fine)
        return tup

    # Impute row-by-row (fast enough; fully vectorizing would add complexity for modest gain)
    num_no_match = 0
    for i in range(N_test):
        key = row_key(i)
        if key not in group_means:
            # If exact combo not seen in TRAIN, just use GLOBAL per-gene means
            fill_vec = global_gene_means
            num_no_match += 1
        else:
            # Use group mean when that gene had >=1 valid contributor; otherwise fall back to global
            gm = group_means[key]
            gc = group_counts[key]
            fill_vec = np.where(gc > 0, gm, global_gene_means)

        cols = np.where(target_mask[i])[0]
        if cols.size:
            imputed_data[i, cols] = fill_vec[cols]
    print(f"Conditional-mean imputation: {num_no_match}/{N_test} test rows had unseen covariate combos; used global means for those.")

    return imputed_data

def _run_one_imputation_gpu(i, args_dict, assigned_gpu, progress_dict=None):
    class NS: pass
    args = NS()
    for k, v in args_dict.items():
        setattr(args, k, v)

    # GPU assignment
    args.gpu_id = assigned_gpu
    if torch.cuda.is_available():
        args.device = f"cuda:{args.gpu_id}"
        torch.cuda.set_device(args.gpu_id)
    else:
        args.device = "cpu"

    # Seeds
    import random
    random.seed(1337 + i)
    np.random.seed(1337 + i)
    torch.manual_seed(1337 + i)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337 + i)

    # Attach progress slot
    if progress_dict is not None:
        args.progress_dict = progress_dict
        args.progress_idx = i
        try:
            progress_dict[i] = 0.0
        except Exception:
            progress_dict = None

    # Prep data + metric
    train_data = SingleCellDataset(args.data_file, split="train", condition_keys=args.cond_keys)
    gene_names = train_data.gene_names
    obs_dict = train_data.get_obs_dict(unique=True)

    # Run
    if args.baseline_model == "magic":
        arr = impute_magic(args, train_data)
    elif args.baseline_model == "scidpm":
        max_arr = np.max(train_data.counts.numpy(), axis=0)
        arr = impute_scidpm(args, max_arr)
        # If scIDPM doesn't expose batch loops, finalize bar:
        if progress_dict is not None:
            try: progress_dict[i] = 1.0
            except: pass
    elif args.baseline_model == "remdm":
        arr = impute_remdm(args)
    elif args.baseline_model == "countsdiff":
        arr = impute_countsdiff(args)
        # (countsdiff already updates per-batch)
    elif args.baseline_model == "mean":
        arr = impute_mean(args, train_data)
        if progress_dict is not None:
            try: progress_dict[i] = 1.0
            except: pass
    elif args.baseline_model == "conditional_mean":
        arr = impute_conditional_means(args, train_data)
        if progress_dict is not None:
            try: progress_dict[i] = 1.0
            except: pass
    else:
        raise ValueError(f"Unknown baseline model {args.baseline_model}")

    return arr

def main():
    args = get_args()

    assert args.baseline_model in {"magic", "gain", "misgan", "forestdiff", "scidpm", "countsdiff", "mean", "conditional_mean", "remdm"}
    assert args.data_type in {"fetus", "heart"}
    assert args.mask_type in {"MCAR", "MAR", "MNAR_high", "MNAR_low"}

    if args.data_type == "fetus":
        args.data_file = f'data/dnadiff/filtered_hca_data.hdf5'
        args.cond_keys = ["disease", "development_day", "batch", "cell_type", "sex"]
        if args.baseline_model == "remdm": # historical naming inconsistency
            args.cond_keys = ['cell_type','disease','sex','batch','development_day']
    elif args.data_type == "heart":
        if getattr(args, "zero_shot", False):
            args.data_file = f'data/dnadiff/zero_shot_heart_data.hdf5'
            args.cond_keys = ["batch", "cell_type"]
            # Match naming pattern used elsewhere
            args.data_type = "zero_shot"
        else:
            args.data_file = f'data/dnadiff/filtered_heart_data.hdf5'
            args.cond_keys = ["batch", "cell_type", "gender", "age"]
    else:
        raise ValueError(f"Unknown data_type {args.data_type}")
    args.mask_file = f'data/dnadiff/random_masks/{args.mask_type}_masks/{args.data_type}_dropout_{args.dropout}.npy'
    os.makedirs(args.save_dir, exist_ok=True)

    # Device here is only used for sequential mode or as a default; workers override it.
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu_id}'
    else:
        args.device = 'cpu'

    # Serialize args so workers can rebuild a Namespace
    args_dict = vars(args).copy()

    # --- Parallel (multi-GPU) or sequential ---
    if args.parallel:
        mp.set_start_method("spawn", force=True)

        gpu_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else list(range(torch.cuda.device_count()))
        if not gpu_list:
            raise RuntimeError("No GPUs available for --parallel. Provide --gpu_ids or set CUDA_VISIBLE_DEVICES.")

        max_workers = min(args.num_workers or len(gpu_list), args.num_imputations)
        print(f"[Parallel] GPUs: {gpu_list} | workers: {max_workers} | imputations: {args.num_imputations}")

        manager = mp.Manager()
        progress = manager.dict()  # i -> float [0,1]

        results = [None] * args.num_imputations
        gpu_assignments = [gpu_list[i % len(gpu_list)] for i in range(args.num_imputations)]

        bars = [
            tqdm(total=100, position=i, leave=True, desc=f"Imp {i} (GPU {gpu_assignments[i]})", dynamic_ncols=True)
            for i in range(args.num_imputations)
        ]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut2i = {ex.submit(_run_one_imputation_gpu, i, args_dict, gpu_assignments[i], progress): i
                     for i in range(args.num_imputations)}

            pending = set(fut2i.keys())
            try:
                while pending:
                    # refresh dataloader-level progress
                    for i, bar in enumerate(bars):
                        frac = float(progress.get(i, 0.0))
                        new_val = int(round(frac * 100))
                        if new_val > bar.n:
                            bar.update(new_val - bar.n)

                    # collect finished futures
                    done_now = {f for f in list(pending) if f.done()}
                    for f in done_now:
                        idx = fut2i[f]
                        results[idx] = f.result()
                        pending.remove(f)

                    time.sleep(0.2)
                # finalize bars
                for bar in bars:
                    if bar.n < 100:
                        bar.update(100 - bar.n)
            finally:
                for bar in bars:
                    bar.close()

    else:
        print(f"[Sequential] Running {args.num_imputations} imputations on device {args.device}")
        results = []
        for i in range(args.num_imputations):
            # Reuse the same worker function with current GPU to keep paths identical
            arr = _run_one_imputation_gpu(i, args_dict, args.gpu_id if torch.cuda.is_available() else -1)
            results.append(arr)

    # Stack and save
    results = np.stack(results, axis=-1)  # [cell, gene, imputation]
    zs_suffix = '_zero_shot' if getattr(args, "zero_shot", False) else ''
    final_path = os.path.join(
        args.save_dir,
        f"{args.baseline_model}_{args.data_type}_results_{args.mask_type}_dropout_{args.dropout}{zs_suffix}_{args.num_imputations}imputations.npy"
    )
    np.save(final_path, results)
    print(f"All imputations saved to {final_path}")
    first_imputation = results[..., 0]
    first_path = f"{args.baseline_model}_{args.data_type}_results_{args.mask_type}_dropout_{args.dropout}{zs_suffix}_1imputation.npy"
    np.save(os.path.join(args.save_dir, first_path), first_imputation)
    print(f"First imputation saved to {os.path.join(args.save_dir, first_path)}")

if __name__ == "__main__":
    main()
