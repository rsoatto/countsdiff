from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_scgpt_imputation import (
    CLS_TOKEN,
    ProgressTracker,
    build_condition_ids,
    choose_indices,
    count_batches,
    impute_test_split,
    initialize_model_from_checkpoint,
    load_checkpoint_metadata,
    load_gene_names,
    map_condition_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained scGPT checkpoint and impute the test split.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--mask-file", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cond-key", action="append", dest="cond_keys", default=[])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--norm-target-sum", type=float, default=None)
    parser.add_argument("--max-test-cells", type=int, default=None)
    parser.add_argument("--progress-path", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    if not args.cond_keys:
        args.cond_keys = list(metadata.get("cond_keys", []))
    if not args.cond_keys:
        raise ValueError(
            "No conditioning keys were provided and the checkpoint metadata does not define them. "
            "Pass --cond-key explicitly or use a checkpoint created by train_scgpt.py."
        )
    if args.norm_target_sum is None:
        args.norm_target_sum = float(metadata.get("norm_target_sum", 10000.0))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    show_console_progress = args.progress_path is None

    with h5py.File(args.data_file, "r") as h5_file:
        gene_names = load_gene_names(h5_file)
        test_group = h5_file["test"]

        test_indices = choose_indices(test_group["counts"].shape[0], args.max_test_cells, args.seed + 23)
        total_progress_units = count_batches(len(test_indices), args.eval_batch_size)
        progress_tracker = ProgressTracker(args.progress_path, total_units=total_progress_units)
        progress_tracker.set_phase("setup", force=True)

        _, record_to_id, num_condition_labels = build_condition_ids(h5_file, args.cond_keys)
        test_condition_ids = map_condition_ids(
            h5_file,
            "test",
            args.cond_keys,
            record_to_id,
            num_condition_labels - 1,
            test_indices,
        )

        model, vocab, gene_ids, _ = initialize_model_from_checkpoint(
            args=args,
            checkpoint_dir=checkpoint_dir,
            gene_names=gene_names,
            num_condition_labels=num_condition_labels,
            device=device,
        )
        progress_tracker.set_phase("initialized model", force=True)

        test_target_mask = np.load(args.mask_file).astype(bool)
        imputed = impute_test_split(
            model=model,
            group=test_group,
            indices=test_indices,
            condition_ids=test_condition_ids,
            batch_size=args.eval_batch_size,
            target_mask_full=test_target_mask,
            gene_ids=gene_ids,
            cls_token_id=vocab[CLS_TOKEN],
            norm_target_sum=args.norm_target_sum,
            device=device,
            progress_tracker=progress_tracker,
            show_console_progress=show_console_progress,
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, imputed)
    progress_tracker.finish("saved")
    print(f"[scGPT] saved imputed array to {output_path}")


if __name__ == "__main__":
    main()
