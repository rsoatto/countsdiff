"""
Parallel evaluator for inference sweeps across multiple GPUs, with optional
per-setting sharding of sample generation.

This script builds the same task grid as eval_inference.py and executes tasks
concurrently using a small worker process per GPU. Each worker loads the model
once on its GPU and pulls tasks from a shared queue. Results are written
per-task (JSON) atomically, enabling safe restarts with --resume.

Additionally, you can split the sample generation for each setting across
multiple workers/GPUs (per-setting sharding). For example, to generate 50k
samples per setting using 10 workers that each generate 5k samples, set
"--setting-workers 10 --num-samples 50000". Each shard saves its samples to
disk; once all shards complete, the main process aggregates samples, computes
metrics for the full set, and writes the final JSON result.

Usage examples:

  # Dynamic assignment of entire settings to GPUs (no per-setting sharding)
  python scripts/eval_inference_pool.py \
    --gpus 0,1,2,3 \
    --run-id XYZ-123 \
    --experiment-name rescaled_remasking_20_guidance \
    --n-steps 400,1000 \
    --guidance-scales 1.0,2.0,3.0 \
    --sigma-methods rescaled \
    --eta-rescales 0.05,0.1 \
    --num-samples 5000 \
    --batch-size 500 \
    --resume

  # Same as above, but split each setting into 10 shards
  python scripts/eval_inference_pool.py \
    --gpus 0,1,2,3 \
    --run-id XYZ-123 \
    --experiment-name rescaled_remasking_20_guidance \
    --n-steps 400,1000 \
    --guidance-scales 1.0,2.0,3.0 \
    --sigma-methods rescaled \
    --eta-rescales 0.05,0.1 \
    --num-samples 50000 \
    --setting-workers 10 \
    --batch-size 500 \
    --resume
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import multiprocessing as mp
import os
import sys
import logging
import warnings
import time
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from countsdiff.generation.generator import CountsdiffGenerator


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


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    # Accept [0,1] or [0,255]; coerce to uint8 [0,255]
    max_val = images.max().item() if torch.is_tensor(images) else float(np.max(images))
    if max_val <= 1.0:
        images = (images * 255.0)
    return images.clamp(0, 255).to(torch.uint8)


def ensure_image_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is a 4D image batch (N,C,H,W). Adds batch dim if needed.
    Raises with a helpful message otherwise.
    """
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Generated tensor must be 4D (N,C,H,W), got shape {tuple(x.shape)}")
    return x


def compute_fid_is(
    real_loader: DataLoader,
    generated: torch.Tensor,
    device: torch.device,
    batch_size: int = 5000,
) -> Tuple[float, float, float]:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    is_metric = InceptionScore(normalize=False).to(device)

    generated = ensure_image_batch(generated)
    gen_loader = DataLoader(TensorDataset(generated), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in gen_loader:
            batch = to_uint8(batch).to(device)
            fid.update(batch, real=False)
            is_metric.update(batch)

        for batch in real_loader:
            if isinstance(batch, (list, tuple)):
                real_images = batch[0]
            else:
                real_images = batch
            real_images = to_uint8(real_images).to(device)
            fid.update(real_images, real=True)

    fid_score = float(fid.compute().item())
    is_mean, is_std = is_metric.compute()
    return float(fid_score), float(is_mean.item()), float(is_std.item())


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
    repaint_num_iters: List[int] = [1],
    repaint_jump: List[int] = [1],
) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    for n_steps, guidance, repaint_num_iters, repaint_jump in itertools.product(steps_list, guidance_list, repaint_num_iters, repaint_jump):
        for method in methods:
            if method == 'none':
                for remask in remask_list:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'none', 'remasking_prob': remask, 'repaint_num_iters': repaint_num_iters, 'repaint_jump': repaint_jump})
            elif method == 'max':
                tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max', 'repaint_num_iters': repaint_num_iters, 'repaint_jump': repaint_jump})
            elif method == 'max_capped':
                for cap in eta_caps:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max_capped', 'eta_cap': cap, 'repaint_num_iters': repaint_num_iters, 'repaint_jump': repaint_jump})
            elif method == 'rescaled':
                for scale in eta_rescales:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'rescaled', 'eta_rescale': scale, 'repaint_num_iters': repaint_num_iters, 'repaint_jump': repaint_jump})
            else:
                raise ValueError(f"Unknown sigma method: {method}")
    return tasks


def base_name_for(spec: Dict[str, Any]) -> str:
    tokens = [f"steps{spec['n_steps']}", f"guid{fmt_val(spec['guidance_scale'])}"]
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


def shard_file_path(shards_output_path: str, base_name: str, part_idx: int) -> str:
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


def run_one_task(
    gen: CountsdiffGenerator,
    trainer,
    spec: Dict[str, Any],
    *,
    num_samples: int,
    batch_size: int,
    device_str: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    method = spec['sigma_method']
    n_steps = int(spec['n_steps'])
    guidance = float(spec['guidance_scale'])

    if method == 'none':
        remask = float(spec['remasking_prob'])
        samples_np = gen.generate_samples(
            num_samples=num_samples,
            n_steps=n_steps,
            device=device_str,
            remasking_prob=remask,
            guidance_scale=guidance,
            batch_size=batch_size,
        )
    elif method == 'max':
        samples_np = gen.generate_samples(
            num_samples=num_samples,
            n_steps=n_steps,
            device=device_str,
            remasking_prob=0.0,
            guidance_scale=guidance,
            batch_size=batch_size,
            sigma_method='max',
            sigma_kwargs={},
        )
    elif method == 'max_capped':
        cap = float(spec['eta_cap'])
        samples_np = gen.generate_samples(
            num_samples=num_samples,
            n_steps=n_steps,
            device=device_str,
            remasking_prob=0.0,
            guidance_scale=guidance,
            batch_size=batch_size,
            sigma_method='max_capped',
            sigma_kwargs={'eta_cap': cap},
        )
    elif method == 'rescaled':
        scale = float(spec['eta_rescale'])
        samples_np = gen.generate_samples(
            num_samples=num_samples,
            n_steps=n_steps,
            device=device_str,
            remasking_prob=0.0,
            guidance_scale=guidance,
            batch_size=batch_size,
            sigma_method='rescaled',
            sigma_kwargs={'eta_rescale': scale},
        )
    else:
        raise ValueError(f"Unknown sigma method: {method}")

    res = {
        'n_steps': n_steps,
        'guidance_scale': guidance,
        'sigma_method': method,
    }
    if method == 'none':
        res['remasking_prob'] = float(spec['remasking_prob'])
    elif method == 'max_capped':
        res['eta_cap'] = float(spec['eta_cap'])
    elif method == 'rescaled':
        res['eta_rescale'] = float(spec['eta_rescale'])

    return samples_np, res


def maybe_save_samples(args, samples_output_path: str, base_name: str, arr: np.ndarray):
    if not args.save_samples:
        return
    count = min(args.save_samples_count, arr.shape[0])
    if count <= 0:
        return
    idx = np.random.choice(arr.shape[0], size=count, replace=False)
    to_save = arr[idx]
    file_path = os.path.join(samples_output_path, f"{base_name}.npz")
    np.savez_compressed(file_path, samples=to_save)
    logging.info(f"Saved {count} samples to {file_path}")


def worker(device_idx: int, device_str: str, task_queue: mp.Queue, progress_queue: mp.Queue, args, exp_output_path: str, results_dir: str, samples_output_path: str, shards_output_path: str, log_file_path: str):
    # Configure per-process logging, warnings, and redirect all output to log file
    configure_logging(log_file_path)
    hard_redirect_to_file(log_file_path)

    if torch.cuda.is_available() and device_str.startswith('cuda:'):
        torch.cuda.set_device(device_idx)
    gen = CountsdiffGenerator(run_id=args.run_id, device=device_str, checkpoint=args.checkpoint)
    trainer = gen.trainer
    if trainer.dataset_type not in ['cifar10', 'celeba']:
        raise ValueError('This evaluation currently supports CIFAR-10 only.')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

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
            shard_path = shard_file_path(shards_output_path, base_name, part_idx)
            if args.resume and os.path.exists(shard_path):
                progress_queue.put(("skip_shard", device_str, base_name, part_idx))
                task_queue.task_done()
                continue

        try:
            if not is_shard:
                progress_queue.put(("start", device_str, base_name))
                samples_np, res = run_one_task(
                    gen, trainer, spec,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    device_str=device_str,
                )

                # Debug: log generated array shape
                logging.info(f"Generated samples for {base_name} with shape {getattr(samples_np, 'shape', None)}")

                maybe_save_samples(args, samples_output_path, base_name, samples_np)

                samples = torch.from_numpy(samples_np).float()
                samples = ensure_image_batch(samples)
                fid, is_mean, is_std = compute_fid_is(trainer.val_loader, samples, device, batch_size=args.batch_size)
                res.update({'fid': fid, 'is_mean': is_mean, 'is_std': is_std})
                progress_queue.put(("done", device_str, base_name, fid, is_mean, is_std))

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
                samples_np, _ = run_one_task(
                    gen, trainer, spec,
                    num_samples=want,
                    batch_size=args.batch_size,
                    device_str=device_str,
                )
                shard_dir = os.path.dirname(shard_file_path(shards_output_path, base_name, part_idx))
                os.makedirs(shard_dir, exist_ok=True)
                tmp = shard_file_path(shards_output_path, base_name, part_idx) + '.tmp'
                np.savez_compressed(tmp, samples=samples_np)
                os.replace(tmp + '.npz', shard_file_path(shards_output_path, base_name, part_idx))
                progress_queue.put(("done_shard", device_str, base_name, part_idx, want))
        except Exception as e:
            logging.exception(f"[GPU {device_str}] ERROR on {base_name}: {e}")
            progress_queue.put(("error", device_str, base_name, str(e)))
        finally:
            task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluator for blackout diffusion (CIFAR-10)")
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs, e.g., "0,1,2". Omit for single GPU.')

    parser.add_argument('--run-id', required=True, help='Neptune run ID to load model and config')
    parser.add_argument('--checkpoint', required=False, help='Run checkpoint')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of samples to generate per setting')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for generation and metrics')

    parser.add_argument('--n-steps', type=str, default='1000', help='Comma-separated list of step counts, e.g. "400,1000"')
    parser.add_argument('--guidance-scales', type=str, default='2.0', help='Comma-separated list, e.g. "1.0,2.0,3.0"')

    parser.add_argument('--sigma-methods', type=str, default='none', help='Methods among: none,max,max_capped,rescaled')
    parser.add_argument('--remasking-probs', type=str, default='0.0', help='Used when sigma-method==none')
    parser.add_argument('--eta-caps', type=str, default='0.05', help='Used for max_capped')
    parser.add_argument('--eta-rescales', type=str, default='0.05', help='Used for rescaled')
    parser.add_argument('--repaint-num-iters', type=str, default='1', help='Comma-separated list of repaint iterations, e.g. "1,2,3"')
    parser.add_argument('--repaint-jumps', type=str, default='1', help='Comma-separated list of repaint jumps, e.g. "1,2,3"')

    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--output-path', type=str, default='data/dnadiff/evals/', help='Root path for results')

    parser.add_argument('--results-subdir', type=str, default='results', help='Where to store per-task JSON results')
    parser.add_argument('--resume', action='store_true', help='Skip tasks whose result file already exists (and shards if present)')
    parser.add_argument('--no-aggregate', action='store_true', help='Do not aggregate results at the end')

    parser.add_argument('--save-samples', action='store_true', help='Save a subset of generated samples for each setting')
    parser.add_argument('--save-samples-count', type=int, default=1000, help='Number of samples to save per setting')
    parser.add_argument('--save-samples-subdir', type=str, default='samples', help='Subdir under experiment output to store sample arrays')

    # Per-setting sharding
    parser.add_argument('--setting-workers', type=int, default=1, help='Split each setting across this many workers; samples per setting are divided across shards')
    parser.add_argument('--shards-subdir', type=str, default='shards', help='Subdir to store per-setting shard sample arrays')
    parser.add_argument('--seed-base', type=int, default=None, help='Optional base seed for deterministic per-shard sampling')
    parser.add_argument('--keep-shards', action='store_true', help='Keep shard files after final aggregation (default: delete)')

    args = parser.parse_args()

    # Parse grids
    steps_list = parse_list(args.n_steps, typ=int) or [1000]
    guidance_list = parse_list(args.guidance_scales, typ=float) or [2.0]
    methods = parse_list(args.sigma_methods, typ=str) or ['none']
    remask_list = parse_list(args.remasking_probs, typ=float) or [0.01]
    eta_caps = parse_list(args.eta_caps, typ=float) or [0.8]
    eta_rescales = parse_list(args.eta_rescales, typ=float) or [0.25]
    repaint_num_iters = parse_list(args.repaint_num_iters, typ=int) or [1]
    repaint_jumps = parse_list(args.repaint_jumps, typ=int) or [1]

    # Paths
    exp_output_path = os.path.join(args.output_path, args.experiment_name)
    os.makedirs(exp_output_path, exist_ok=True)
    results_dir = os.path.join(exp_output_path, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    samples_output_path = os.path.join(exp_output_path, args.save_samples_subdir)
    if args.save_samples:
        os.makedirs(samples_output_path, exist_ok=True)
    shards_output_path = os.path.join(exp_output_path, args.shards_subdir)
    if args.setting_workers > 1:
        os.makedirs(shards_output_path, exist_ok=True)
    log_file_path = os.path.join(exp_output_path, 'eval.log')

    # Configure logging in the main process (file only). Keep stdout for tqdm.
    configure_logging(log_file_path)

    # Build and optionally filter tasks for resume
    base_tasks = build_tasks(steps_list, guidance_list, methods, remask_list, eta_caps, eta_rescales, repaint_num_iters, repaint_jumps)
    if args.resume:
        def done(spec):
            return os.path.exists(os.path.join(results_dir, f"{base_name_for(spec)}.json"))
        base_tasks = [t for t in base_tasks if not done(t)]
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"Resume mode: {len(base_tasks)} base tasks remaining.")
    else:
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"Total base tasks: {len(base_tasks)}")

    # Expand into shard tasks if requested
    if args.setting_workers > 1:
        tasks: List[Dict[str, Any]] = []
        for spec in base_tasks:
            for part_idx in range(args.setting_workers):
                s = dict(spec)
                s['_part_idx'] = part_idx
                s['_nparts'] = args.setting_workers
                tasks.append(s)
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"Per-setting sharding: expanded to {len(tasks)} shard tasks")
    else:
        tasks = base_tasks

    # GPU devices to use
    if args.gpus is None:
        if torch.cuda.is_available():
            devices = ["cuda:0"]
        else:
            devices = ["cpu"]
    else:
        gpus = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
        devices = [f"cuda:{i}" for i in gpus]

    # Start workers
    task_queue: mp.Queue = mp.JoinableQueue()
    for spec in tasks:
        task_queue.put(spec)
    for _ in devices:
        task_queue.put(None)

    # Progress queue for worker→main updates
    progress_queue: mp.Queue = mp.Queue()

    procs: List[mp.Process] = []
    for dev in devices:
        idx = 0 if dev == 'cpu' else int(dev.split(':')[1])
        p = mp.Process(target=worker, args=(idx, dev, task_queue, progress_queue, args, exp_output_path, results_dir, samples_output_path, shards_output_path, log_file_path))
        p.daemon = False
        p.start()
        procs.append(p)
        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(f"Started worker on {dev}, pid={p.pid}")
    # Drive a single tqdm progress bar from main
    total_tasks = len(tasks)
    completed = 0
    bar = tqdm(total=total_tasks, desc='Eval', dynamic_ncols=True)
    while completed < total_tasks:
        try:
            msg = progress_queue.get(timeout=0.5)
        except Exception:
            continue
        if not msg:
            continue
        kind = msg[0]
        if kind == 'start':
            _, dev, base = msg
            bar.set_postfix_str(f"{dev} {base}")
        elif kind == 'start_shard':
            _, dev, base, part_idx, want = msg
            bar.set_postfix_str(f"{dev} {base} part{part_idx:03d} ({want})")
        elif kind == 'skip':
            _, dev, base = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[Skip] {dev} {base}")
        elif kind == 'skip_shard':
            _, dev, base, part_idx = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[Skip shard] {dev} {base} part{part_idx:03d}")
        elif kind == 'done':
            _, dev, base, fid, is_mean, is_std = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[{dev}] {base} -> FID={fid:.3f} IS={is_mean:.3f}±{is_std:.3f}")
        elif kind == 'done_shard':
            _, dev, base, part_idx, want = msg
            completed += 1
            bar.update(1)
            tqdm.write(f"[{dev}] {base} part{part_idx:03d} saved ({want})")
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
        # Iterate base (non-sharded) specs and aggregate their shards
        base_specs = build_tasks(steps_list, guidance_list, methods, remask_list, eta_caps, eta_rescales, repaint_num_iters, repaint_jumps)
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
                    shard_paths.append(os.path.join(shard_dir, f"part_{i:03d}.npz"))
            if not shard_paths or not all(os.path.exists(p) for p in shard_paths):
                _tqdm.write(f"[Agg warn] Missing shards for {base_name}; skipping")
                continue

            # Load shards and concatenate
            parts = []
            ok = True
            for sp in shard_paths:
                try:
                    with np.load(sp) as d:
                        parts.append(d['samples'])
                except Exception as e:
                    logging.warning(f"Failed to load shard {sp}: {e}")
                    ok = False
                    break
            if not ok:
                continue
            all_samples_np = np.concatenate(parts, axis=0)
            if all_samples_np.shape[0] > args.num_samples:
                all_samples_np = all_samples_np[:args.num_samples]

            # Compute metrics against real validation images
            metrics_device = torch.device(devices[0] if torch.cuda.is_available() else 'cpu')
            gen_for_metrics = CountsdiffGenerator(run_id=args.run_id, device=str(metrics_device), checkpoint=args.checkpoint)
            samples = torch.from_numpy(all_samples_np).float()
            fid, is_mean, is_std = compute_fid_is(gen_for_metrics.trainer.val_loader, samples, metrics_device, batch_size=args.batch_size)

            res = {
                'n_steps': int(spec['n_steps']),
                'guidance_scale': float(spec['guidance_scale']),
                'sigma_method': spec['sigma_method'],
                'fid': float(fid),
                'is_mean': float(is_mean),
                'is_std': float(is_std),
            }
            if spec['sigma_method'] == 'none':
                res['remasking_prob'] = float(spec['remasking_prob'])
            elif spec['sigma_method'] == 'max_capped':
                res['eta_cap'] = float(spec['eta_cap'])
            elif spec['sigma_method'] == 'rescaled':
                res['eta_rescale'] = float(spec['eta_rescale'])

            tmp_path = result_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(res, f, indent=2)
            os.replace(tmp_path, result_path)
            _tqdm.write(f"[Agg] {base_name} -> FID={fid:.3f} IS={is_mean:.3f}±{is_std:.3f}")

            # Optionally save a subset of full samples and clean up shards
            if args.save_samples:
                maybe_save_samples(args, samples_output_path, base_name, all_samples_np)
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
        aggregate_path = os.path.join(exp_output_path, 'inference-sweep.json')
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
