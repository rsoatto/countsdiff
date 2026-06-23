from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scgpt_imputation import (
    CLS_TOKEN,
    ProgressTracker,
    build_condition_ids,
    count_batches,
    initialize_model_for_training,
    load_gene_names,
    map_condition_ids,
    save_checkpoint_dir,
    set_seed,
    train_model,
    choose_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an scGPT model and save a reusable checkpoint.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-mode", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--pretrained-dir", default=None, help="Required if --init-mode=pretrained")
    parser.add_argument("--t-dir", type=str, default=None)
    parser.add_argument("--cond-key", action="append", dest="cond_keys", default=[])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-mask-ratio", type=float, default=0.4)
    parser.add_argument("--norm-target-sum", type=float, default=10000.0)
    parser.add_argument("--embsize", type=int, default=128)
    parser.add_argument("--d-hid", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-train-cells", type=int, default=None)
    parser.add_argument("--max-val-cells", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--progress-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.init_mode == "pretrained" and args.pretrained_dir is None:
        raise ValueError("--pretrained-dir is required when --init-mode=pretrained")

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

        train_condition_ids_full, record_to_id, num_condition_labels = build_condition_ids(h5_file, args.cond_keys)
        train_condition_ids = train_condition_ids_full[train_indices]
        val_condition_ids = map_condition_ids(
            h5_file,
            "val",
            args.cond_keys,
            record_to_id,
            num_condition_labels - 1,
            val_indices,
        )

        model, vocab, gene_ids, model_config = initialize_model_for_training(
            args=args,
            gene_names=gene_names,
            num_condition_labels=num_condition_labels,
            device=device,
        )
        progress_tracker.set_phase("initialized model", force=True)

        cls_token_id = vocab[CLS_TOKEN]
        _, best_state, best_val = train_model(
            model=model,
            train_group=train_group,
            val_group=val_group,
            train_indices=train_indices,
            val_indices=val_indices,
            train_condition_ids=train_condition_ids,
            val_condition_ids=val_condition_ids,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            lr=args.lr,
            mask_ratio=args.train_mask_ratio,
            gene_ids=gene_ids,
            cls_token_id=cls_token_id,
            norm_target_sum=args.norm_target_sum,
            device=device,
            seed=args.seed,
            progress_tracker=progress_tracker,
            show_console_progress=show_console_progress,
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    save_checkpoint_dir(
        checkpoint_dir=checkpoint_dir,
        model_state=best_state,
        vocab=vocab,
        model_config=model_config,
        metadata={
            "cond_keys": list(args.cond_keys),
            "data_file": args.data_file,
            "init_mode": args.init_mode,
            "norm_target_sum": float(args.norm_target_sum),
            "run_name": args.run_name,
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "best_val": float(best_val),
        },
    )
    progress_tracker.finish("saved checkpoint")
    print(f"[scGPT] saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()
