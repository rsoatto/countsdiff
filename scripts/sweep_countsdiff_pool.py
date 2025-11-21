"""
"""

from __future__ import annotations

import warnings

# Suppress only DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



import argparse
import itertools
import json
import random
import multiprocessing as mp
import os
import sys
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm
from queue import Empty

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import linalg

from countsdiff.generation.generator import CountsdiffImputer
from countsdiff.utils.metrics import scFID



# -----------------------------------------------------------------------------
# Logging and warning control
# -----------------------------------------------------------------------------

class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level

    def write(self, buf: str):
        if not buf:
            return
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line)

    def flush(self):
        pass


def configure_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger()
    # Avoid duplicate handlers if called multiple times in child processes
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s [%(processName)s %(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # Route warnings to logging and suppress DeprecationWarnings on console
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def redirect_std_to_logger() -> Tuple[Any, Any]:
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = _StreamToLogger(stderr_logger, logging.ERROR)
    return orig_stdout, orig_stderr


def hard_redirect_to_file(log_path: str) -> None:
    """Redirect OS-level stdout/stderr and Python sys streams to a file.
    Ensures even C/CUDA prints are captured and not shown on console.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, 'a', buffering=1)
    try:
        os.dup2(log_f.fileno(), 1)
        os.dup2(log_f.fileno(), 2)
    except Exception:
        pass
    sys.stdout = log_f
    sys.stderr = log_f


def parse_list(arg: Optional[str], typ=float) -> Optional[List[Any]]:
    if arg is None:
        return None
    items = [x.strip() for x in arg.split(',') if x.strip()]
    return [typ(x) for x in items]


def fmt_val(v: Any) -> str:
    s = f"{v}" if isinstance(v, float) else str(v)
    return s.replace('-', 'm').replace('.', 'p')


def build_tasks(
    steps_list: List[int],
    guidance_list: List[float],
    methods: List[str],
    remask_list: List[float],
    eta_caps: List[float],
    eta_rescales: List[float],
    repaint_num_iters: Optional[int] = None,
    repaint_jumps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for n_steps, guidance, repaint_num_iters, repaint_jumps in itertools.product(steps_list, guidance_list, repaint_num_iters, repaint_jumps):
        for method in methods:
            if method == 'none':
                for remask in remask_list:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'none', 'remasking_prob': remask, 'repaint_num_iters': repaint_num_iters, 'repaint_jumps': repaint_jumps})
            elif method == 'max':
                tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max', 'repaint_num_iters': repaint_num_iters, 'repaint_jumps': repaint_jumps})
            elif method == 'max_capped':
                for cap in eta_caps:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max_capped', 'eta_cap': cap, 'repaint_num_iters': repaint_num_iters, 'repaint_jumps': repaint_jumps})
            elif method == 'rescaled':
                for scale in eta_rescales:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'rescaled', 'eta_rescale': scale, 'repaint_num_iters': repaint_num_iters, 'repaint_jumps': repaint_jumps})
            else:
                raise ValueError(f"Unknown sigma method: {method}")
    return tasks


def base_name_for(spec: Dict[str, Any]) -> str:
    tokens = [f"steps{spec['n_steps']}", f"guid{fmt_val(spec['guidance_scale'])}", f"repaint{spec['repaint_num_iters']}j{spec['repaint_jumps']}"]
    if spec['sigma_method'] == 'none':
        tokens += ["none", f"remask{fmt_val(spec['remasking_prob'])}"]
    elif spec['sigma_method'] == 'max':
        tokens += ["max"]
    elif spec['sigma_method'] == 'max_capped':
        tokens += ["maxcapped", f"cap{fmt_val(spec['eta_cap'])}"]
    elif spec['sigma_method'] == 'rescaled':
        tokens += ["rescaled", f"res{fmt_val(spec['eta_rescale'])}"]
    return "_".join(tokens)


def shard_sizes(total: int, num_parts: int) -> List[int]:
    base = total // num_parts
    rem = total % num_parts
    return [base + (1 if i < rem else 0) for i in range(num_parts)]


def shard_metrics_path(shards_output_path: str, base_name: str, part_idx: int) -> str:
    return os.path.join(shards_output_path, base_name, f"part_{part_idx:03d}_metrics.npz")

def shard_samples_path(shards_output_path: str, base_name: str, part_idx: int) -> str:
    return os.path.join(shards_output_path, base_name, f"part_{part_idx:03d}.npz")


def set_seed_all(seed: int) -> None:
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)


def compute_rmse_r2(imputed: torch.Tensor, ground_truth: torch.Tensor, impute_mask: torch.Tensor) -> Tuple[float, float]:
    imputed_vals = torch.masked_select(imputed, impute_mask.bool())
    actual_vals = torch.masked_select(ground_truth, impute_mask.bool())
    rmse = torch.sqrt(torch.mean((imputed_vals - actual_vals) ** 2)).item()
    ss_res = torch.sum((actual_vals - imputed_vals) ** 2)
    ss_tot = torch.sum((actual_vals - torch.mean(actual_vals)) ** 2)
    # Handle degenerate case gracefully
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)).item())
    return float(rmse), r2

def fid_from_latents(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_fake, sigma_fake = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    sum_sq_diff = float(np.sum((mu_real - mu_fake) ** 2.0))
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = sum_sq_diff + float(np.trace(sigma_real + sigma_fake - 2.0 * covmean))
    return float(fid)


def run_one_task_impute(
    imputer: CountsdiffImputer,
    trainer,
    spec: Dict[str, Any],
    *,
    num_cells: int,
    batch_size: int,
    dropout_ratio: float,
    device_str: str,
    scfid_metric: Optional[scFID] = None,
) -> Dict[str, Any]:
    """Run a single imputation task and compute metrics.

    Returns a dict with metrics and the spec fields used.
    """
    method = spec['sigma_method']
    n_steps = int(spec['n_steps'])
    guidance = float(spec['guidance_scale'])

    # Update imputer hyperparameters for this spec
    if method == 'none':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method=None,
            sigma_kwargs=None,
            batch_size=batch_size,
        )
    elif method == 'max':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max',
            sigma_kwargs={},
            batch_size=batch_size,
        )
    elif method == 'max_capped':
        cap = float(spec['eta_cap'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max_capped',
            sigma_kwargs={'eta_cap': cap},
            batch_size=batch_size,
        )
    elif method == 'rescaled':
        scale = float(spec['eta_rescale'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='rescaled',
            sigma_kwargs={'eta_rescale': scale},
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown sigma method: {method}")

    # Build one batch from val dataset
    val_ds = trainer.val_loader.dataset
    # Ensure we don't exceed dataset size
    num_cells = min(num_cells, len(val_ds))
    dl = DataLoader(val_ds, batch_size=num_cells, shuffle=True)
    batch = next(iter(dl))
    counts, labels, missing_masks = batch
    valid_mask = (~missing_masks).to(imputer.device).bool()

    counts = counts.to(imputer.device)
    labels = [label.to(imputer.device) for label in labels] if isinstance(labels, list) else labels.to(imputer.device)
    missing_masks = missing_masks.to(imputer.device)
    impute_mask = (torch.rand_like(counts) < float(dropout_ratio)) & valid_mask
    masked = counts * (~impute_mask).float().to(imputer.device)


    with torch.no_grad():
        logging.info(
            f"Imputing {impute_mask.sum().item()} of {valid_mask.sum().item()} valid values"
        )
        imputed_np = imputer.impute_data(
            counts=masked,
            valid_mask=valid_mask,
            impute_mask=impute_mask,
            labels=labels,
        )

    imputed = torch.from_numpy(imputed_np) if isinstance(imputed_np, np.ndarray) else imputed_np
    imputed = imputed.to(counts.device)
    # Restore known entries
    imputed[~impute_mask] = counts[~impute_mask]

    # Metrics
    rmse, r2 = compute_rmse_r2(imputed, counts, impute_mask)

    # scFID using trainer's metric (already initialized for scrna)
    scfid_value = None
    if scfid_metric is not None:
        scfid_metric = scfid_metric.to(imputer.device).eval()
        scfid_metric.reset()
        cov_df = val_ds.build_covariate_df(labels)
        scfid_metric.update(counts.cpu().numpy(), cov_df, True)
        scfid_metric.update(imputed.detach().cpu().numpy(), cov_df, False)
        scfid_value = float(scfid_metric.compute().item())
        scfid_metric.reset()

    res: Dict[str, Any] = {
        'n_steps': n_steps,
        'guidance_scale': guidance,
        'sigma_method': method,
        'dropout_ratio': float(dropout_ratio),
        'rmse': float(rmse),
        'r2': float(r2),
    }
    if scfid_value is not None:
        res['scfid'] = float(scfid_value)

    if method == 'none':
        res['remasking_prob'] = float(spec['remasking_prob'])
    elif method == 'max_capped':
        res['eta_cap'] = float(spec['eta_cap'])
    elif method == 'rescaled':
        res['eta_rescale'] = float(spec['eta_rescale'])

    return res


def compute_imputation_shard(
    imputer: CountsdiffImputer,
    trainer,
    spec: Dict[str, Any],
    *,
    num_cells: int,
    batch_size: int,
    dropout_ratio: float,
    device_str: str,
    scfid_metric: Optional[scFID] = None,
) -> Dict[str, Any]:
    """Compute shard-level contributions for metrics and return details.

    Returns dict with:
      - sse, n_imputed, sum_actual, sum_actual2
      - real_latents, fake_latents (np arrays) if scfid_metric provided
    """
    method = spec['sigma_method']
    n_steps = int(spec['n_steps'])
    guidance = float(spec['guidance_scale'])

    # Update hyperparameters
    if method == 'none':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method=None,
            sigma_kwargs=None,
            batch_size=batch_size,
        )
    elif method == 'max':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max',
            sigma_kwargs={},
            batch_size=batch_size,
        )
    elif method == 'max_capped':
        cap = float(spec['eta_cap'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max_capped',
            sigma_kwargs={'eta_cap': cap},
            batch_size=batch_size,
        )
    elif method == 'rescaled':
        scale = float(spec['eta_rescale'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='rescaled',
            sigma_kwargs={'eta_rescale': scale},
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown sigma method: {method}")

    val_ds = trainer.val_loader.dataset
    num_cells = min(num_cells, len(val_ds))
    dl = DataLoader(val_ds, batch_size=num_cells, shuffle=True)
    counts, labels, missing_masks = next(iter(dl))
    counts = counts.to(imputer.device)
    labels = [label.to(imputer.device) for label in labels] if isinstance(labels, list) else labels.to(imputer.device)
    missing_masks = missing_masks.to(imputer.device)
    valid_mask = (~missing_masks).to(imputer.device).bool()
    impute_mask = (torch.rand_like(counts) < float(dropout_ratio)) & valid_mask
    masked = counts * (~impute_mask).float().to(imputer.device)

    with torch.no_grad():
        imputed_np = imputer.impute_data(
            counts=masked,
            valid_mask=valid_mask,
            impute_mask=impute_mask,
            labels=labels,
        )
    imputed = torch.from_numpy(imputed_np) if isinstance(imputed_np, np.ndarray) else imputed_np
    imputed = imputed.to(counts.device)
    imputed[~impute_mask] = counts[~impute_mask]

    # Contributions for RMSE/R2
    imputed_vals = torch.masked_select(imputed, impute_mask.bool()).float()
    actual_vals = torch.masked_select(counts, impute_mask.bool()).float()
    diff = (imputed_vals - actual_vals)
    sse = float(torch.sum(diff * diff).item())
    n_imputed = int(imputed_vals.numel())
    sum_actual = float(torch.sum(actual_vals).item())
    sum_actual2 = float(torch.sum(actual_vals * actual_vals).item())

    real_latents = None
    fake_latents = None
    if scfid_metric is not None:
        scfid_metric = scfid_metric.to(imputer.device).eval()
        scfid_metric.reset()
        cov_df = val_ds.build_covariate_df(labels)
        scfid_metric.update(counts.cpu().numpy(), cov_df, True)
        scfid_metric.update(imputed.detach().cpu().numpy(), cov_df, False)
        # Extract latent arrays
        real_latents = torch.cat(scfid_metric.real_latents, dim=0).cpu().numpy()
        fake_latents = torch.cat(scfid_metric.fake_latents, dim=0).cpu().numpy()
        scfid_metric.reset()

    return {
        'sse': sse,
        'n_imputed': n_imputed,
        'sum_actual': sum_actual,
        'sum_actual2': sum_actual2,
        'real_latents': real_latents,
        'fake_latents': fake_latents,
    }


def compute_imputation_shard_samples(
    imputer: CountsdiffImputer,
    trainer,
    spec: Dict[str, Any],
    *,
    num_cells: int,
    batch_size: int,
    dropout_ratio: float,
    device_str: str,
) -> Dict[str, Any]:
    """Generate imputed samples for a shard and return arrays to save.

    Returns dict with:
      - counts: np.ndarray [cells x genes]
      - imputed: np.ndarray [cells x genes] (with known entries restored)
      - impute_mask: np.ndarray [cells x genes] bool, True where imputed
      - labels_list: List[np.ndarray] length = num_condition_keys, each [cells]
    """
    method = spec['sigma_method']
    n_steps = int(spec['n_steps'])
    guidance = float(spec['guidance_scale'])

    # Update hyperparameters for this spec
    if method == 'none':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method=None,
            sigma_kwargs=None,
            batch_size=batch_size,
        )
    elif method == 'max':
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max',
            sigma_kwargs={},
            batch_size=batch_size,
        )
    elif method == 'max_capped':
        cap = float(spec['eta_cap'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='max_capped',
            sigma_kwargs={'eta_cap': cap},
            batch_size=batch_size,
        )
    elif method == 'rescaled':
        scale = float(spec['eta_rescale'])
        imputer.update_hyperparameters(
            n_steps=n_steps,
            device=device_str,
            guidance_scale=guidance,
            sigma_method='rescaled',
            sigma_kwargs={'eta_rescale': scale},
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown sigma method: {method}")

    val_ds = trainer.val_loader.dataset
    num_cells = min(num_cells, len(val_ds))
    dl = DataLoader(val_ds, batch_size=num_cells, shuffle=True)
    counts, labels, missing_masks = next(iter(dl))

    counts = counts.to(imputer.device)
    labels = [label.to(imputer.device) for label in labels] if isinstance(labels, list) else labels.to(imputer.device)
    missing_masks = missing_masks.to(imputer.device)
    valid_mask = (~missing_masks).to(imputer.device).bool()
    impute_mask = (torch.rand_like(counts) < float(dropout_ratio)) & valid_mask
    masked = counts * (~impute_mask).float().to(imputer.device)

    with torch.no_grad():
        imputed_np = imputer.impute_data(
            counts=masked,
            valid_mask=valid_mask,
            impute_mask=impute_mask,
            labels=labels,
        )

    imputed = torch.from_numpy(imputed_np) if isinstance(imputed_np, np.ndarray) else imputed_np
    imputed = imputed.to(counts.device)
    # Restore known entries
    imputed[~impute_mask] = counts[~impute_mask]

    # Convert to numpy for saving
    counts_np = counts.detach().cpu().numpy()
    imputed_np = imputed.detach().cpu().numpy()
    impute_mask_np = impute_mask.detach().cpu().numpy().astype(np.bool_)
    if isinstance(labels, list):
        labels_list = [lbl.detach().cpu().numpy() for lbl in labels]
    else:
        labels_list = [labels.detach().cpu().numpy()]

    return {
        'counts': counts_np,
        'imputed': imputed_np,
        'impute_mask': impute_mask_np,
        'labels_list': labels_list,
    }


def worker(device_idx: int, device_str: str, task_queue: mp.Queue, progress_queue: mp.Queue, args, exp_output_path: str, results_dir: str, samples_output_path: str, shards_output_path: str, log_file_path: str):
    # Configure per-process logging, warnings, and redirect all output to log file
    configure_logging(log_file_path)
    hard_redirect_to_file(log_file_path)

    if torch.cuda.is_available() and device_str.startswith('cuda:'):
        torch.cuda.set_device(device_idx)

    # Initialize imputer on this device
    imputer = CountsdiffImputer(run_id=args.run_id, device=device_str, checkpoint=args.checkpoint)
    trainer = imputer.generator.trainer
    if trainer.dataset_type != 'scrna':
        raise ValueError('This evaluation supports scRNA-seq (dataset_type==scrna). Check the run-id and config.')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    # Prepare scFID metric if possible (build from dataset + provided scvi model)
    scfid_metric = None
    try:
        val_ds = trainer.val_loader.dataset
        gene_names = getattr(val_ds, 'gene_names', None)
        if gene_names is not None and args.scvi_model_path is not None and os.path.exists(args.scvi_model_path):
            # categorical_covariates expects full list of possible categories per covariate
            categorical_covariates = val_ds.get_obs_dict(unique=True)
            scfid_metric = scFID(gene_names=gene_names, categorical_covariates=categorical_covariates, feature_model_path=args.scvi_model_path)
    except Exception as e:
        logging.warning(f"scFID initialization failed: {e}")

    while True:
        spec = task_queue.get()
        if spec is None:
            task_queue.task_done()
            break

        base_name = base_name_for(spec)
        is_shard = ('_nparts' in spec and spec['_nparts'] is not None)
        if not is_shard:
            result_path = os.path.join(results_dir, f"{base_name}.json")
            if args.resume and os.path.exists(result_path):
                progress_queue.put(("skip", device_str, base_name))
                task_queue.task_done()
                continue
        else:
            part_idx = int(spec['_part_idx'])
            nparts = int(spec['_nparts'])
            sizes = shard_sizes(args.num_samples, nparts)
            want = sizes[part_idx]
            shard_path = shard_samples_path(shards_output_path, base_name, part_idx)
            if args.resume and os.path.exists(shard_path):
                progress_queue.put(("skip_shard", device_str, base_name, part_idx))
                task_queue.task_done()
                continue

        try:
            if not is_shard:
                progress_queue.put(("start", device_str, base_name))
                res = run_one_task_impute(
                    imputer, trainer, spec,
                    num_cells=args.num_samples,
                    batch_size=args.batch_size,
                    dropout_ratio=args.dropout_ratio,
                    device_str=device_str,
                    scfid_metric=scfid_metric,
                )

                # Log and save
                logging.info(f"Computed metrics for {base_name}: {res}")
                progress_queue.put(("done", device_str, base_name, res.get('rmse'), res.get('r2'), res.get('scfid')))

                tmp_path = result_path + '.tmp'
                with open(tmp_path, 'w') as f:
                    json.dump(res, f, indent=2)
                os.replace(tmp_path, result_path)
            else:
                # Optional deterministic seed per shard
                if args.seed_base is not None:
                    seed = int(args.seed_base) + (hash(base_name) % 10_000_000) + int(part_idx)
                    set_seed_all(seed)

                progress_queue.put(("start_shard", device_str, base_name, part_idx, want))
                shard_samples = compute_imputation_shard_samples(
                    imputer, trainer, spec,
                    num_cells=want,
                    batch_size=args.batch_size,
                    dropout_ratio=args.dropout_ratio,
                    device_str=device_str,
                )
                shard_dir = os.path.dirname(shard_samples_path(shards_output_path, base_name, part_idx))
                os.makedirs(shard_dir, exist_ok=True)
                tmp = shard_samples_path(shards_output_path, base_name, part_idx) + '.tmp.npz'
                # Save counts, imputed, mask, and labels
                save_kwargs = {
                    'counts': shard_samples['counts'].astype(np.float32),
                    'imputed': shard_samples['imputed'].astype(np.float32),
                    'impute_mask': shard_samples['impute_mask'].astype(np.bool_),
                    'n_labels': np.array(len(shard_samples['labels_list']), dtype=np.int64),
                }
                for i, lab in enumerate(shard_samples['labels_list']):
                    save_kwargs[f'label_{i}'] = lab.astype(np.int64)
                np.savez_compressed(tmp, **save_kwargs)
                os.replace(tmp, shard_samples_path(shards_output_path, base_name, part_idx))
                progress_queue.put(("done_shard", device_str, base_name, part_idx, want))

        except Exception as e:
            logging.exception(f"[GPU {device_str}] ERROR on {base_name}: {e}")
            progress_queue.put(("error", device_str, base_name, str(e)))
        finally:
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluator for scRNA-seq imputation (CountsdiffImputer)")
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs, e.g., "0,1,2". Omit for single GPU.')

    parser.add_argument('--run-id', required=True, help='Neptune run ID to load model and config')
    parser.add_argument('--checkpoint', required=False, help='Run checkpoint')
    parser.add_argument('--num-samples', type=int, default=1024, help='Number of cells per setting to evaluate')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for imputation calls (internal batching)')
    parser.add_argument('--dropout-ratio', type=float, default=0.4, help='Random dropout ratio for imputation mask (applied on valid positions only)')

    parser.add_argument('--n-steps', type=str, default='100', help='Comma-separated list of step counts, e.g. "100,400"')
    parser.add_argument('--guidance-scales', type=str, default='1.0', help='Comma-separated list, e.g. "1.0,2.0"')

    parser.add_argument('--sigma-methods', type=str, default='rescaled', help='Methods among: none,max,max_capped,rescaled')
    parser.add_argument('--remasking-probs', type=str, default='0.0', help='Used when sigma-method==none')
    parser.add_argument('--eta-caps', type=str, default='0.05', help='Used for max_capped')
    parser.add_argument('--eta-rescales', type=str, default='0.005', help='Used for rescaled')
    parser.add_argument('--repaint-num-iters', type=str, default='1', help='Comma-separated list of repaint iterations, e.g. "1,3,5"')
    parser.add_argument('--repaint-jumps', type=str, default='1', help='Comma-separated list of repaint jumps, e.g. "1,3,5"')

    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--output-path', type=str, default='data/dnadiff/evals/', help='Root path for results')
    parser.add_argument('--scvi-model-path', type=str, default='data/dnadiff/2024-02-12-scvi-homo-sapiens/scvi.model', help='Path to scvi-tools model directory for scFID features')

    parser.add_argument('--results-subdir', type=str, default='results', help='Where to store per-task JSON results')
    parser.add_argument('--resume', action='store_true', help='Skip tasks whose result file already exists')
    parser.add_argument('--no-aggregate', action='store_true', help='Do not aggregate results at the end')

    # Per-setting sharding
    parser.add_argument('--setting-workers', type=int, default=1, help='Split each setting across this many workers; metrics aggregated across shards')
    parser.add_argument('--shards-subdir', type=str, default='shards', help='Subdir to store per-setting shard arrays')
    parser.add_argument('--seed-base', type=int, default=None, help='Optional base seed for deterministic per-shard sampling')
    parser.add_argument('--keep-shards', action='store_true', help='Keep shard files after final aggregation (default: delete)')

    args = parser.parse_args()

    # Parse grids
    steps_list = parse_list(args.n_steps, typ=int) or [100]
    guidance_list = parse_list(args.guidance_scales, typ=float) or [1.0]
    methods = parse_list(args.sigma_methods, typ=str) or ['rescaled']
    remask_list = parse_list(args.remasking_probs, typ=float) or [0.0]
    eta_caps = parse_list(args.eta_caps, typ=float) or [0.05]
    eta_rescales = parse_list(args.eta_rescales, typ=float) or [0.005]
    repaint_num_iters = parse_list(args.repaint_num_iters, typ=int) or [1]
    repaint_jumps = parse_list(args.repaint_jumps, typ=int) or [1]

    # Paths
    exp_output_path = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(exp_output_path, exist_ok=True)
    results_dir = os.path.join(exp_output_path, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    samples_output_path = os.path.join(exp_output_path, 'samples')  # unused but keep param for worker signature parity
    shards_output_path = os.path.join(exp_output_path, args.shards_subdir)
    os.makedirs(shards_output_path, exist_ok=True)

    # Build full list of (base) tasks
    base_specs = build_tasks(steps_list, guidance_list, methods, remask_list, eta_caps, eta_rescales, repaint_num_iters, repaint_jumps)

    # GPU assignment
    if args.gpus is None:
        devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    else:
        devices = [f"cuda:{x.strip()}" for x in args.gpus.split(',') if x.strip()]
        if not devices:
            devices = ["cpu"]

    # Queue tasks (support per-setting sharding)
    task_queue: mp.JoinableQueue = mp.JoinableQueue()
    progress_queue: mp.Queue = mp.Queue()
    if args.setting_workers > 1:
        for spec in base_specs:
            for part_idx in range(args.setting_workers):
                shard_spec = dict(spec)
                shard_spec['_nparts'] = args.setting_workers
                shard_spec['_part_idx'] = part_idx
                task_queue.put(shard_spec)
    else:
        for spec in base_specs:
            task_queue.put(spec)
    # Sentinel per worker will be added after workers start

    # Spawn one worker process per device
    procs: List[mp.Process] = []
    for i, dev in enumerate(devices):
        log_file_path = os.path.join(exp_output_path, f"worker_{i}.log")
        p = mp.Process(
            target=worker,
            args=(i, dev, task_queue, progress_queue, args, exp_output_path, results_dir, samples_output_path, shards_output_path, log_file_path),
            name=f"impute@{dev}"
        )
        p.start()
        procs.append(p)

    for _ in procs:
        task_queue.put(None)

    # Monitor progress
    total = len(base_specs) * (args.setting_workers if args.setting_workers > 1 else 1)
    completed = 0
    bar = tqdm(total=total, desc="Imputation eval tasks")
    while completed < total:
        msg = progress_queue.get()
        kind = msg[0]
        if kind == 'skip':
            _, dev, base = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[Skip {dev}] {base}")
        elif kind == 'start':
            _, dev, base = msg
            tqdm.write(f"[{dev}] Start {base}")
        elif kind == 'done':
            _, dev, base, rmse, r2, scfid = msg
            completed += 1
            bar.update(1)
            if scfid is not None:
                tqdm.write(f"[{dev}] {base} -> RMSE={rmse:.4f} R2={r2:.4f} scFID={scfid:.3f}")
            else:
                tqdm.write(f"[{dev}] {base} -> RMSE={rmse:.4f} R2={r2:.4f}")
        elif kind == 'start_shard':
            _, dev, base, part_idx, want = msg
            tqdm.write(f"[{dev}] {base} part{part_idx:03d} -> {want} cells")
        elif kind == 'done_shard':
            _, dev, base, part_idx, want = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[{dev}] {base} part{part_idx:03d} saved ({want})")
        elif kind == 'skip_shard':
            _, dev, base, part_idx = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[Skip shard {dev}] {base} part{int(part_idx):03d}")
        elif kind == 'error':
            _, dev, base, err = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[Error {dev}] {base}: {err}")
    bar.close()

    # Ensure workers exit
    for p in procs:
        p.join()

    # If we sharded per-setting, aggregate shards into final per-setting results
    if args.setting_workers > 1:
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write("Aggregating per-setting shards into final results...")
        imputer_for_metrics = CountsdiffImputer(run_id=args.run_id, device=str(devices[0] if torch.cuda.is_available() else 'cpu'), checkpoint=args.checkpoint)
        val_ds_for_metrics = imputer_for_metrics.generator.trainer.val_loader.dataset
        gene_names = getattr(val_ds_for_metrics, 'gene_names', None)
        if gene_names is not None and args.scvi_model_path is not None and os.path.exists(args.scvi_model_path):
            categorical_covariates = val_ds_for_metrics.get_obs_dict(unique=True)
            scfid_metric = scFID(gene_names=gene_names, categorical_covariates=categorical_covariates, feature_model_path=args.scvi_model_path)

        for spec in base_specs:
            base_name = base_name_for(spec)
            result_path = os.path.join(results_dir, f"{base_name}.json")
            if args.resume and os.path.exists(result_path):
                _tqdm.write(f"[Agg skip] {base_name} (result exists)")
                continue

            shard_dir = os.path.join(shards_output_path, base_name)
            shard_paths = []
            if os.path.isdir(shard_dir):
                for i in range(args.setting_workers):
                    shard_paths.append(shard_samples_path(shards_output_path, base_name, i))
            if not shard_paths or not all(os.path.exists(p) for p in shard_paths):
                _tqdm.write(f"[Agg warn] Missing shards for {base_name}; skipping")
                continue

            # Load shards and concatenate samples
            counts_parts: List[np.ndarray] = []
            imputed_parts: List[np.ndarray] = []
            mask_parts: List[np.ndarray] = []
            labels_parts: Optional[List[List[np.ndarray]]] = None

            ok = True
            for sp in shard_paths:
                try:
                    with np.load(sp, allow_pickle=False) as d:
                        counts_parts.append(d['counts'])
                        imputed_parts.append(d['imputed'])
                        mask_parts.append(d['impute_mask'].astype(np.bool_))
                        n_labels = int(d['n_labels']) if 'n_labels' in d else 0
                        shard_labels = []
                        for i in range(n_labels):
                            shard_labels.append(d[f'label_{i}'])
                        if labels_parts is None:
                            labels_parts = [[] for _ in range(n_labels)]
                        if n_labels != len(labels_parts):
                            raise ValueError("Inconsistent number of label arrays across shards")
                        for i, arr in enumerate(shard_labels):
                            labels_parts[i].append(arr)
                except Exception as e:
                    logging.warning(f"Failed to load shard {sp}: {e}")
                    ok = False
                    break
            if not ok:
                continue

            all_counts = np.concatenate(counts_parts, axis=0)
            all_imputed = np.concatenate(imputed_parts, axis=0)
            all_mask = np.concatenate(mask_parts, axis=0).astype(np.bool_)
            if all_counts.shape[0] > args.num_samples:
                all_counts = all_counts[:args.num_samples]
                all_imputed = all_imputed[:args.num_samples]
                all_mask = all_mask[:args.num_samples]
                if labels_parts is not None:
                    labels_parts = [np.concatenate(x, axis=0)[:args.num_samples] for x in labels_parts]
            else:
                if labels_parts is not None:
                    labels_parts = [np.concatenate(x, axis=0) for x in labels_parts]

            # Compute RMSE and R2 on masked entries
            imputed_vals = all_imputed[all_mask].astype(np.float64)
            actual_vals = all_counts[all_mask].astype(np.float64)
            n_total = max(1, imputed_vals.size)
            sse_total = float(np.sum((imputed_vals - actual_vals) ** 2.0))
            rmse = float(np.sqrt(sse_total / n_total))
            mean_actual = float(np.mean(actual_vals)) if n_total > 0 else 0.0
            ss_tot = float(np.sum((actual_vals - mean_actual) ** 2.0))
            r2 = float(1.0 - (sse_total / (ss_tot + 1e-12)))

            # Compute scFID from aggregated samples if possible
            scfid_value = None
            try:
                cov_df = val_ds_for_metrics.build_covariate_df(labels_parts)
                scfid_metric.update(all_counts, cov_df, True)
                scfid_metric.update(all_imputed, cov_df, False)
                scfid_value = float(scfid_metric.compute().item())
                scfid_metric.reset()
            except Exception as e:
                logging.warning(f"scFID aggregation failed for {base_name}: {e}")

            res = {
                'n_steps': int(spec['n_steps']),
                'guidance_scale': float(spec['guidance_scale']),
                'sigma_method': spec['sigma_method'],
                'dropout_ratio': float(args.dropout_ratio),
                'rmse': float(rmse),
                'r2': float(r2),
            }
            if scfid_value is not None:
                res['scfid'] = float(scfid_value)
            if spec['sigma_method'] == 'none':
                res['remasking_prob'] = float(spec['remasking_prob'])
            elif spec['sigma_method'] == 'max_capped':
                res['eta_cap'] = float(spec['eta_cap'])
            elif spec['sigma_method'] == 'rescaled':
                res['eta_rescale'] = float(spec['eta_rescale'])
            if 'repaint_num_iters' in spec and spec['repaint_num_iters'] is not None:
                res['repaint_num_iters'] = int(spec['repaint_num_iters'])
            if 'repaint_jumps' in spec and spec['repaint_jumps'] is not None:
                res['repaint_jump'] = int(spec['repaint_jumps'])

            tmp_path = result_path + '.tmp.npz'
            with open(tmp_path, 'w') as f:
                json.dump(res, f, indent=2)
            os.replace(tmp_path, result_path)
            _tqdm.write(f"[Agg] {base_name} -> RMSE={rmse:.4f} R2={r2:.4f}" + (f" scFID={scfid_value:.3f}" if scfid_value is not None else ""))

            # Optionally clean up shards
            if not args.keep_shards:
                for sp in shard_paths:
                    try:
                        os.remove(sp)
                    except Exception:
                        pass
                try:
                    if os.path.isdir(shard_dir) and not os.listdir(shard_dir):
                        os.rmdir(shard_dir)
                except Exception:
                    pass

    if not args.no_aggregate:
        aggregate_path = os.path.join(exp_output_path, 'imputation-sweep.json')
        aggregated: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(results_dir)):
            if not name.endswith('.json'):
                continue
            with open(os.path.join(results_dir, name), 'r') as f:
                try:
                    aggregated.append(json.load(f))
                except Exception as e:
                    logging.warning(f"Skipping malformed result {name}: {e}")
        with open(aggregate_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"Aggregated {len(aggregated)} results into {aggregate_path}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
