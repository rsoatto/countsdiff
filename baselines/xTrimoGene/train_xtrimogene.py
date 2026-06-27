from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from xtrimogene_runner import (
    ProgressTracker,
    choose_indices,
    count_batches,
    initialize_model_for_training,
    load_gene_names,
    save_checkpoint_dir,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an xTrimoGene model and save a reusable checkpoint.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=9766)
    parser.add_argument("--value-mask-prob", type=float, default=0.3)
    parser.add_argument("--zero-mask-prob", type=float, default=0.03)
    parser.add_argument("--replace-prob", type=float, default=0.8)
    parser.add_argument("--random-token-prob", type=float, default=0.1)
    parser.add_argument("--norm-target-sum", type=float, default=10000.0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--max-train-cells", type=int, default=None)
    parser.add_argument("--max-val-cells", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--progress-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    show_console_progress = args.progress_path is None

    with h5py.File(args.data_file, "r") as h5_file:
        gene_names = load_gene_names(h5_file)
        train_group = h5_file["train"]
        val_group = h5_file["val"]

        train_indices = choose_indices(train_group["counts"].shape[0], args.max_train_cells, args.seed + 11)
        val_indices = choose_indices(val_group["counts"].shape[0], args.max_val_cells, args.seed + 17)
        total_progress_units = (
            args.epochs * count_batches(len(train_indices), args.batch_size)
            + args.epochs * count_batches(len(val_indices), args.eval_batch_size)
        )
        progress_tracker = ProgressTracker(args.progress_path, total_units=total_progress_units)
        progress_tracker.set_phase("setup", force=True)

        model, model_config = initialize_model_for_training(gene_names, device)
        progress_tracker.set_phase("initialized model", force=True)

        _, best_state, best_val = train_model(
            model=model,
            train_group=train_group,
            val_group=val_group,
            train_indices=train_indices,
            val_indices=val_indices,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            value_mask_prob=args.value_mask_prob,
            zero_mask_prob=args.zero_mask_prob,
            replace_prob=args.replace_prob,
            random_token_prob=args.random_token_prob,
            norm_target_sum=args.norm_target_sum,
            gradient_clip_val=args.gradient_clip_val,
            model_config=model_config,
            device=device,
            seed=args.seed,
            progress_tracker=progress_tracker,
            show_console_progress=show_console_progress,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    save_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        model_state=best_state,
        model_config=model_config,
        metadata={
            "architecture": "xtrimogene_3m",
            "data_file": args.data_file,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "best_val": float(best_val),
            "norm_target_sum": float(args.norm_target_sum),
            "run_name": args.run_name,
            "gene_names": list(gene_names),
        },
    )
    progress_tracker.finish("saved checkpoint")
    print(f"[xTrimoGene] saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
