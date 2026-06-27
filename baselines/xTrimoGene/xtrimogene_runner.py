from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

try:
    from .xtrimogene_model import MASK_TOKEN_VALUE, build_model_from_config, build_xtrimogene_3m_model, get_encoder_decoder_data
except ImportError:
    from xtrimogene_model import MASK_TOKEN_VALUE, build_model_from_config, build_xtrimogene_3m_model, get_encoder_decoder_data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_indices(total: int, max_cells: Optional[int], seed: int) -> np.ndarray:
    if max_cells is None or max_cells >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=max_cells, replace=False).astype(np.int64))


def load_gene_names(h5_file: h5py.File) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in h5_file["gene_names"][:]]


def count_batches(num_examples: int, batch_size: int) -> int:
    if num_examples <= 0:
        return 0
    return (num_examples + batch_size - 1) // batch_size


class ProgressTracker:
    def __init__(self, progress_path: Optional[str], total_units: int) -> None:
        self.progress_path = Path(progress_path) if progress_path is not None else None
        self.total_units = max(int(total_units), 1)
        self.completed_units = 0
        self.last_fraction = -1.0
        self.last_phase: Optional[str] = None
        self.last_write_time = 0.0

    def _write(self, phase: str, *, force: bool = False) -> None:
        if self.progress_path is None:
            return

        fraction = min(max(self.completed_units / self.total_units, 0.0), 1.0)
        now = time.monotonic()
        should_write = force
        should_write = should_write or phase != self.last_phase
        should_write = should_write or (fraction - self.last_fraction) >= 1e-3
        should_write = should_write or (now - self.last_write_time) >= 0.25
        if not should_write:
            return

        payload = {
            "fraction": fraction,
            "phase": phase,
            "completed_units": self.completed_units,
            "total_units": self.total_units,
        }
        tmp_path = self.progress_path.with_suffix(self.progress_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self.progress_path)
        self.last_fraction = fraction
        self.last_phase = phase
        self.last_write_time = now

    def set_phase(self, phase: str, *, force: bool = False) -> None:
        self._write(phase, force=force)

    def advance(self, phase: str, units: int = 1) -> None:
        self.completed_units = min(self.total_units, self.completed_units + int(units))
        self._write(phase)

    def finish(self, phase: str = "done") -> None:
        self.completed_units = self.total_units
        self._write(phase, force=True)


def iter_index_batches(indices: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64)
    if shuffle and ordered.size > 1:
        rng = np.random.default_rng(seed)
        ordered = ordered[rng.permutation(len(ordered))]
    for start in range(0, len(ordered), batch_size):
        yield np.sort(ordered[start:start + batch_size])


def normalize_from_masked_profile(
    raw_counts: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    norm_target_sum: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = raw_counts.astype(np.float32, copy=True)
    valid = valid_mask.astype(bool, copy=False)
    observed = raw.copy()
    observed[~valid] = 0.0
    observed[target_mask] = 0.0
    retained_library = observed.sum(axis=1, keepdims=True).astype(np.float32)
    retained_library = np.clip(retained_library, 1.0, None)

    normalized_full = np.log1p((raw * valid.astype(np.float32)) * (float(norm_target_sum) / retained_library))
    normalized_full[~valid] = 0.0
    return normalized_full.astype(np.float32), observed.astype(np.float32), retained_library[:, 0]


def sample_training_target_mask(
    raw_counts: np.ndarray,
    valid_mask: np.ndarray,
    value_mask_prob: float,
    zero_mask_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    valid = valid_mask.astype(bool, copy=False)
    nonzero_positions = (raw_counts > 0) & valid
    zero_positions = (raw_counts <= 0) & valid
    nonzero_mask = rng.random(raw_counts.shape) < float(value_mask_prob)
    zero_mask = rng.random(raw_counts.shape) < float(zero_mask_prob)
    return (nonzero_positions & nonzero_mask) | (zero_positions & zero_mask)


def apply_replacement_strategy(
    normalized_full: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    replace_prob: float,
    random_token_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    masked_input = normalized_full.astype(np.float32, copy=True)
    active_mask = target_mask & valid_mask
    if not np.any(active_mask):
        return masked_input

    draws = rng.random(masked_input.shape)
    replace_mask = active_mask & (draws < float(replace_prob))
    random_mask = active_mask & (draws >= float(replace_prob)) & (draws < float(replace_prob + random_token_prob))
    masked_input[replace_mask] = MASK_TOKEN_VALUE

    if np.any(random_mask):
        pool = normalized_full[(valid_mask) & (normalized_full > 0)]
        if pool.size == 0:
            pool = np.array([0.0], dtype=np.float32)
        masked_input[random_mask] = rng.choice(pool, size=int(random_mask.sum()), replace=True).astype(np.float32)
    return masked_input


def initialize_model_for_training(gene_names: Sequence[str], device: torch.device):
    model, model_config = build_xtrimogene_3m_model(len(gene_names))
    model = model.to(device)
    return model, model_config


def save_checkpoint_dir(
    checkpoint_dir: Path,
    model_state: Dict[str, torch.Tensor],
    model_config: Dict[str, object],
    metadata: Dict[str, object],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model_state}, checkpoint_dir / "best_model.pt")
    (checkpoint_dir / "config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_checkpoint_metadata(checkpoint_dir: Path) -> Dict[str, object]:
    return json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))


def load_checkpoint_config(checkpoint_dir: Path) -> Dict[str, object]:
    return json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))


def initialize_model_from_checkpoint(
    checkpoint_dir: Path,
    gene_names: Sequence[str],
    device: torch.device,
):
    model_config = load_checkpoint_config(checkpoint_dir)
    metadata = load_checkpoint_metadata(checkpoint_dir)
    if int(model_config["seq_len"]) != len(gene_names):
        raise ValueError(
            f"Checkpoint expects seq_len={model_config['seq_len']} genes, but the current dataset has {len(gene_names)} genes."
        )
    saved_gene_names = metadata.get("gene_names")
    if isinstance(saved_gene_names, list) and list(saved_gene_names) != list(gene_names):
        raise ValueError("Checkpoint gene ordering does not match the current dataset.")

    model = build_model_from_config(model_config).to(device)
    state = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model, model_config, metadata


def _run_model(
    model: torch.nn.Module,
    input_values: np.ndarray,
    observed_counts: np.ndarray,
    model_config: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    input_tensor = torch.as_tensor(input_values, device=device, dtype=torch.float32)
    observed_tensor = torch.as_tensor(observed_counts, device=device, dtype=torch.float32)
    (
        encoder_data,
        encoder_position_gene_ids,
        encoder_padding,
        encoder_labels,
        decoder_data,
        decoder_padding,
        _,
        _,
        decoder_position_gene_ids,
    ) = get_encoder_decoder_data(input_tensor, observed_tensor, model_config)

    return model(
        encoder_data,
        encoder_padding,
        encoder_position_gene_ids,
        encoder_labels,
        decoder_data,
        False,
        None,
        decoder_position_gene_ids,
        decoder_padding,
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr_value: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_value


def train_model(
    *,
    model: torch.nn.Module,
    train_group: h5py.Group,
    val_group: h5py.Group,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    value_mask_prob: float,
    zero_mask_prob: float,
    replace_prob: float,
    random_token_prob: float,
    norm_target_sum: float,
    gradient_clip_val: float,
    model_config: Dict[str, object],
    device: torch.device,
    seed: int,
    progress_tracker: ProgressTracker,
    show_console_progress: bool,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if warmup_steps > 0:
        _set_optimizer_lr(optimizer, lr / float(warmup_steps))

    train_batches = count_batches(len(train_indices), batch_size)
    val_batches = count_batches(len(val_indices), eval_batch_size)
    total_optimizer_steps = epochs * max(1, (train_batches + max(grad_accum_steps, 1) - 1) // max(grad_accum_steps, 1))
    if warmup_steps > total_optimizer_steps:
        print(
            f"[xTrimoGene] warning: warmup_steps={warmup_steps} exceeds total optimizer steps={total_optimizer_steps}; "
            "the learning rate will remain in warmup for the full run."
        )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    optimizer_steps = 0

    for epoch in tqdm(range(1, epochs + 1), desc="Training"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        train_bar = None
        if show_console_progress:
            train_bar = tqdm(total=train_batches, desc=f"train {epoch}/{epochs}", leave=True, dynamic_ncols=True)

        train_rng = np.random.default_rng(seed + epoch * 1009)
        for batch_idx, batch_indices in enumerate(
            iter_index_batches(train_indices, batch_size, shuffle=True, seed=seed + epoch * 37),
            start=1,
        ):
            raw_counts = train_group["counts"][batch_indices].astype(np.float32, copy=False)
            valid_mask = ~train_group["missingness_mask"][batch_indices].astype(bool, copy=False)
            target_mask = sample_training_target_mask(raw_counts, valid_mask, value_mask_prob, zero_mask_prob, train_rng)

            phase = f"train {epoch}/{epochs} [{batch_idx}/{train_batches}]"
            if not np.any(target_mask):
                progress_tracker.advance(phase)
                if train_bar is not None:
                    train_bar.update(1)
                continue

            normalized_full, observed_counts, _ = normalize_from_masked_profile(
                raw_counts,
                target_mask,
                valid_mask,
                norm_target_sum,
            )
            masked_input = apply_replacement_strategy(
                normalized_full,
                target_mask,
                valid_mask,
                replace_prob,
                random_token_prob,
                train_rng,
            )
            prediction = _run_model(model, masked_input, observed_counts, model_config, device)
            target_tensor = torch.as_tensor(normalized_full, device=device, dtype=torch.float32)
            target_mask_tensor = torch.as_tensor(target_mask, device=device, dtype=torch.bool)
            loss = F.mse_loss(prediction[target_mask_tensor], target_tensor[target_mask_tensor])
            train_losses.append(float(loss.item()))
            (loss / max(grad_accum_steps, 1)).backward()

            should_step = (batch_idx % max(grad_accum_steps, 1) == 0) or (batch_idx == train_batches)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                warmup_scale = 1.0 if warmup_steps <= 0 else min(optimizer_steps / float(warmup_steps), 1.0)
                _set_optimizer_lr(optimizer, lr * warmup_scale)

            progress_tracker.advance(phase)
            if train_bar is not None:
                train_bar.update(1)
                if train_losses:
                    train_bar.set_postfix(loss=f"{np.mean(train_losses):.4f}", refresh=False)

        if train_bar is not None:
            train_bar.close()

        model.eval()
        val_losses = []
        val_bar = None
        if show_console_progress:
            val_bar = tqdm(total=val_batches, desc=f"val {epoch}/{epochs}", leave=True, dynamic_ncols=True)

        val_rng = np.random.default_rng(seed + epoch * 2039)
        with torch.no_grad():
            for batch_idx, batch_indices in enumerate(
                iter_index_batches(val_indices, eval_batch_size, shuffle=False, seed=seed),
                start=1,
            ):
                raw_counts = val_group["counts"][batch_indices].astype(np.float32, copy=False)
                valid_mask = ~val_group["missingness_mask"][batch_indices].astype(bool, copy=False)
                target_mask = sample_training_target_mask(raw_counts, valid_mask, value_mask_prob, zero_mask_prob, val_rng)
                phase = f"val {epoch}/{epochs} [{batch_idx}/{val_batches}]"
                if not np.any(target_mask):
                    progress_tracker.advance(phase)
                    if val_bar is not None:
                        val_bar.update(1)
                    continue

                normalized_full, observed_counts, _ = normalize_from_masked_profile(
                    raw_counts,
                    target_mask,
                    valid_mask,
                    norm_target_sum,
                )
                masked_input = apply_replacement_strategy(
                    normalized_full,
                    target_mask,
                    valid_mask,
                    replace_prob,
                    random_token_prob,
                    val_rng,
                )
                prediction = _run_model(model, masked_input, observed_counts, model_config, device)
                target_tensor = torch.as_tensor(normalized_full, device=device, dtype=torch.float32)
                target_mask_tensor = torch.as_tensor(target_mask, device=device, dtype=torch.bool)
                loss = F.mse_loss(prediction[target_mask_tensor], target_tensor[target_mask_tensor])
                val_losses.append(float(loss.item()))

                progress_tracker.advance(phase)
                if val_bar is not None:
                    val_bar.update(1)
                    if val_losses:
                        val_bar.set_postfix(loss=f"{np.mean(val_losses):.4f}", refresh=False)

        if val_bar is not None:
            val_bar.close()

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_state, best_val


def impute_test_split(
    *,
    model: torch.nn.Module,
    group: h5py.Group,
    indices: np.ndarray,
    batch_size: int,
    target_mask_full: np.ndarray,
    norm_target_sum: float,
    model_config: Dict[str, object],
    device: torch.device,
    progress_tracker: ProgressTracker,
    show_console_progress: bool,
) -> np.ndarray:
    model.eval()
    outputs = []
    total_batches = count_batches(len(indices), batch_size)
    test_bar = None
    if show_console_progress:
        test_bar = tqdm(total=total_batches, desc="test", leave=True, dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, batch_indices in enumerate(
            iter_index_batches(indices, batch_size, shuffle=False, seed=0),
            start=1,
        ):
            raw_counts = group["counts"][batch_indices].astype(np.float32, copy=False)
            valid_mask = ~group["missingness_mask"][batch_indices].astype(bool, copy=False)
            target_mask = target_mask_full[batch_indices].astype(bool, copy=False) & valid_mask

            normalized_full, observed_counts, retained_library = normalize_from_masked_profile(
                raw_counts,
                target_mask,
                valid_mask,
                norm_target_sum,
            )
            masked_input = normalized_full.copy()
            masked_input[target_mask] = MASK_TOKEN_VALUE

            prediction = _run_model(model, masked_input, observed_counts, model_config, device)
            prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
            prediction_counts = np.expm1(prediction_np) * (retained_library[:, None] / float(norm_target_sum))
            prediction_counts = np.clip(prediction_counts, 0.0, None)

            imputed = raw_counts.astype(np.float32, copy=True)
            imputed[target_mask] = prediction_counts[target_mask]
            outputs.append(imputed)

            phase = f"test [{batch_idx}/{total_batches}]"
            progress_tracker.advance(phase)
            if test_bar is not None:
                test_bar.update(1)

    if test_bar is not None:
        test_bar.close()
    return np.concatenate(outputs, axis=0)
