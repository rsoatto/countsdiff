"""
Evaluate blackout diffusion inference hyperparameters (CIFAR-10) by FID and Inception Score.

Sweeps over:
- number of steps (iterations)
- guidance scale
- remasking schedules (constant, max, max_capped, rescaled)

Requires a trained run (Neptune run_id) so CountsdiffGenerator can load the model and data.
"""

import argparse
import itertools
import json
import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from countsdiff.generation.generator import CountsdiffGenerator


def to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Convert tensor images to uint8 [0,255]. Accepts values in [0,255] or [0,1]."""
    if images.max() <= 1.0:
        images = (images * 255.0)
    return images.clamp(0, 255).byte()


def compute_fid_is(
    real_loader: DataLoader,
    generated: torch.Tensor,
    device: torch.device,
    batch_size: int = 500,
) -> Tuple[float, float, float]:
    """Compute FID and Inception Score for generated images against real_loader images."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    is_metric = InceptionScore(normalize=False).to(device)

    # Update fake
    gen_loader = DataLoader(TensorDataset(generated), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in gen_loader:
            batch = to_uint8(batch).to(device)
            fid.update(batch, real=False)
            is_metric.update(batch)

        # Update real
        for batch in real_loader:
            # real_loader may return (images, labels) or images only
            if isinstance(batch, (list, tuple)):
                real_images = batch[0]
            else:
                real_images = batch
            real_images = to_uint8(real_images).to(device)
            fid.update(real_images, real=True)

    fid_score = float(fid.compute().item())
    is_mean, is_std = is_metric.compute()
    is_mean = float(is_mean.item())
    is_std = float(is_std.item())
    return fid_score, is_mean, is_std


def parse_list(arg: Optional[str], typ=float) -> Optional[List[Any]]:
    if arg is None:
        return None
    items = [x.strip() for x in arg.split(',') if x.strip()]
    return [typ(x) for x in items]


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference hyperparameters for blackout diffusion (CIFAR-10)")
    parser.add_argument('--run-id', required=True, help='Neptune run ID to load model and config')
    parser.add_argument('--checkpoint', required=False, help='Run checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of samples to generate per setting')
    parser.add_argument('--batch-size', type=int, default=500, help='Batch size for generation and metrics')

    parser.add_argument('--n-steps', type=str, default='1000', help='Comma-separated list of step counts, e.g. "400,1000"')
    parser.add_argument('--guidance-scales', type=str, default='2.0', help='Comma-separated list of guidance scales, e.g. "1.0,2.0,3.0"')

    parser.add_argument('--sigma-methods', type=str, default='none',
                        help='Comma-separated sigma schedule methods among: none,max,max_capped,rescaled')
    parser.add_argument('--remasking-probs', type=str, default='0.0',
                        help='Comma-separated constant remasking probs (used when sigma-method==none)')
    parser.add_argument('--eta-caps', type=str, default='0.05', help='Comma-separated eta_cap values for max_capped')
    parser.add_argument('--eta-rescales', type=str, default='0.05', help='Comma-separated eta_rescale values for rescaled')

    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')

    parser.add_argument('--output-path', type=str, default='data/dnadiff/evals/', help='Optional path to save results under this root')

    # Parallelization/sharding + checkpointing
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of shards (processes) splitting the sweep')
    parser.add_argument('--shard-id', type=int, default=0, help='This shard index in [0, num-shards)')
    parser.add_argument('--results-subdir', type=str, default='results', help='Subdirectory under experiment output to store per-task JSON results')
    parser.add_argument('--resume', action='store_true', help='Skip tasks whose result file already exists')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate per-task results into a single JSON and exit')

    # Saving generated samples
    parser.add_argument('--save-samples', action='store_true', help='Save a subset of generated samples for each setting')
    parser.add_argument('--save-samples-count', type=int, default=1000, help='Number of samples to save per setting')
    parser.add_argument('--save-samples-subdir', type=str, default='samples', help='Subdirectory under experiment output to store sample arrays')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build generator (loads model, data, and EMA)
    gen = CountsdiffGenerator(run_id=args.run_id, device=args.device, checkpoint=args.checkpoint)
    trainer = gen.trainer
    if trainer.dataset_type != 'cifar10':
        raise ValueError('This evaluation script currently supports CIFAR-10 only.')

    # Parse grids
    steps_list = parse_list(args.n_steps, typ=int) or [1000]
    guidance_list = parse_list(args.guidance_scales, typ=float) or [2.0]
    methods = parse_list(args.sigma_methods, typ=str) or ['none']
    remask_list = parse_list(args.remasking_probs, typ=float) or [0.01]
    eta_caps = parse_list(args.eta_caps, typ=float) or [0.8]
    eta_rescales = parse_list(args.eta_rescales, typ=float) or [0.25]

    results: List[Dict[str, Any]] = []

    # Paths
    exp_output_path = f"{args.output_path}/{args.experiment_name}"
    os.makedirs(exp_output_path, exist_ok=True)
    results_dir = os.path.join(exp_output_path, args.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    samples_output_path = os.path.join(exp_output_path, args.save_samples_subdir)
    if args.save_samples:
        os.makedirs(samples_output_path, exist_ok=True)

    def fmt_val(v: Any) -> str:
        """Format hyperparameter values for filenames (e.g., 2.0 -> 2p0, -0.05 -> m0p05)."""
        if isinstance(v, float):
            s = f"{v}"
        else:
            s = str(v)
        s = s.replace('-', 'm').replace('.', 'p')
        return s

    def maybe_save_samples(base_name: str, arr: np.ndarray):
        if not args.save_samples:
            return
        count = min(args.save_samples_count, arr.shape[0])
        if count <= 0:
            return
        idx = np.random.choice(arr.shape[0], size=count, replace=False)
        to_save = arr[idx]
        file_path = os.path.join(samples_output_path, f"{base_name}.npz")
        np.savez_compressed(file_path, samples=to_save)
        print(f"  -> Saved {count} samples to {file_path}")
    # Build the task grid
    tasks: List[Dict[str, Any]] = []
    for n_steps, guidance in itertools.product(steps_list, guidance_list):
        for method in methods:
            if method == 'none':
                for remask in remask_list:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'none', 'remasking_prob': remask})
            elif method == 'max':
                tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max'})
            elif method == 'max_capped':
                for cap in eta_caps:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'max_capped', 'eta_cap': cap})
            elif method == 'rescaled':
                for scale in eta_rescales:
                    tasks.append({'n_steps': n_steps, 'guidance_scale': guidance, 'sigma_method': 'rescaled', 'eta_rescale': scale})
            else:
                raise ValueError(f"Unknown sigma method: {method}")

    # Optional aggregation mode (no compute), to be used after parallel runs
    if args.aggregate:
        aggregate_path = os.path.join(exp_output_path, 'inference-sweep.json')
        aggregated: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(results_dir)):
            if not name.endswith('.json'):
                continue
            with open(os.path.join(results_dir, name), 'r') as f:
                try:
                    aggregated.append(json.load(f))
                except Exception as e:
                    print(f"Warning: skipping malformed result {name}: {e}")
        with open(aggregate_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"Aggregated {len(aggregated)} results into {aggregate_path}")
        return

    # Determine this shard's slice
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError('--shard-id must satisfy 0 <= shard_id < num_shards')

    # Execute tasks belonging to this shard
    for idx, spec in enumerate(tasks):
        if idx % args.num_shards != args.shard_id:
            continue

        # Base name and result path
        base_tokens = [
            f"steps{spec['n_steps']}",
            f"guid{fmt_val(spec['guidance_scale'])}",
        ]
        if spec['sigma_method'] == 'none':
            base_tokens += ["none", f"remask{fmt_val(spec['remasking_prob'])}"]
        elif spec['sigma_method'] == 'max':
            base_tokens += ["max"]
        elif spec['sigma_method'] == 'max_capped':
            base_tokens += ["maxcapped", f"cap{fmt_val(spec['eta_cap'])}"]
        elif spec['sigma_method'] == 'rescaled':
            base_tokens += ["rescaled", f"res{fmt_val(spec['eta_rescale'])}"]
        base_name = "_".join(base_tokens)

        result_path = os.path.join(results_dir, f"{base_name}.json")
        if args.resume and os.path.exists(result_path):
            print(f"[Skip] Existing result for {base_name}")
            continue

        # Run generation
        method = spec['sigma_method']
        n_steps = int(spec['n_steps'])
        guidance = float(spec['guidance_scale'])
        print(f"[Shard {args.shard_id}/{args.num_shards}] Eval {base_name}")

        if method == 'none':
            remask = float(spec['remasking_prob'])
            samples_np = gen.generate_samples(
                num_samples=args.num_samples,
                n_steps=n_steps,
                device=args.device,
                remasking_prob=remask,
                guidance_scale=guidance,
                batch_size=args.batch_size,
            )
        elif method == 'max':
            samples_np = gen.generate_samples(
                num_samples=args.num_samples,
                n_steps=n_steps,
                device=args.device,
                remasking_prob=0.0,
                guidance_scale=guidance,
                batch_size=args.batch_size,
                sigma_method='max',
                sigma_kwargs={},
            )
        elif method == 'max_capped':
            cap = float(spec['eta_cap'])
            samples_np = gen.generate_samples(
                num_samples=args.num_samples,
                n_steps=n_steps,
                device=args.device,
                remasking_prob=0.0,
                guidance_scale=guidance,
                batch_size=args.batch_size,
                sigma_method='max_capped',
                sigma_kwargs={'eta_cap': cap},
            )
        elif method == 'rescaled':
            scale = float(spec['eta_rescale'])
            samples_np = gen.generate_samples(
                num_samples=args.num_samples,
                n_steps=n_steps,
                device=args.device,
                remasking_prob=0.0,
                guidance_scale=guidance,
                batch_size=args.batch_size,
                sigma_method='rescaled',
                sigma_kwargs={'eta_rescale': scale},
            )
        else:
            raise ValueError(f"Unknown sigma method: {method}")

        maybe_save_samples(base_name=base_name, arr=samples_np)

        # Compute metrics and write per-task JSON atomically
        samples = torch.from_numpy(samples_np).float()
        fid, is_mean, is_std = compute_fid_is(trainer.val_loader, samples, device, batch_size=args.batch_size)
        res = {
            'n_steps': n_steps,
            'guidance_scale': guidance,
            'sigma_method': method,
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
        }
        if method == 'none':
            res['remasking_prob'] = float(spec['remasking_prob'])
        elif method == 'max_capped':
            res['eta_cap'] = float(spec['eta_cap'])
        elif method == 'rescaled':
            res['eta_rescale'] = float(spec['eta_rescale'])

        print(f"  -> FID={fid:.3f} IS={is_mean:.3f}±{is_std:.3f}")

        tmp_path = result_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(res, f, indent=2)
        os.replace(tmp_path, result_path)
        results.append(res)

    # If running single-shard, also aggregate into top-level JSON
    if args.num_shards == 1:
        output_json = f"{exp_output_path}/inference-sweep.json"
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_json}")


if __name__ == '__main__':
    main()
