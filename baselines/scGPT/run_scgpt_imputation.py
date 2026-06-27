from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm


PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
EOC_TOKEN = "<eoc>"
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, EOC_TOKEN]
MASK_VALUE = -1.0
PAD_VALUE = -2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run the scGPT imputation baseline.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--mask-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-mode", choices=["scratch", "pretrained"], required=True)
    parser.add_argument("--pretrained-dir", type=str, default=None)
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
    parser.add_argument("--max-test-cells", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--progress-path", type=str, default=None)
    parser.add_argument("--save-checkpoint-dir", type=str, default=None)
    return parser.parse_args()


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


def load_gene_names(h5_file: h5py.File) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in h5_file["gene_names"][:]]


def load_condition_arrays(h5_file: h5py.File, split: str, cond_keys: Sequence[str], indices: Optional[np.ndarray] = None) -> List[np.ndarray]:
    if not cond_keys:
        return []
    group = h5_file[split]
    split_group = {key.lower(): value for key, value in group.items()}
    arrays: List[np.ndarray] = []
    for key in cond_keys:
        value_key = f"{key.lower()}_values"
        if value_key not in split_group:
            available = sorted(split_group.keys())
            raise KeyError(
                f"Condition key '{key}' resolved to '{value_key}', but that dataset was not found in "
                f"split '{split}'. Available datasets: {available}"
            )
        raw = split_group[value_key]
        arr = raw[:] if indices is None else raw[indices]
        arrays.append(arr)
    return arrays


def build_condition_ids(h5_file: h5py.File, cond_keys: Sequence[str]) -> Tuple[np.ndarray, Dict[bytes, int], int]:
    if not cond_keys:
        train_size = h5_file["train"]["counts"].shape[0]
        return np.zeros(train_size, dtype=np.int64), {}, 1

    arrays = load_condition_arrays(h5_file, "train", cond_keys)
    structured = np.rec.fromarrays(arrays)
    _, inverse = np.unique(structured, return_inverse=True)
    unique_records = np.unique(structured)
    record_to_id = {record.tobytes(): idx for idx, record in enumerate(unique_records)}
    unk_id = len(unique_records)
    return inverse.astype(np.int64), record_to_id, unk_id + 1


def map_condition_ids(
    h5_file: h5py.File,
    split: str,
    cond_keys: Sequence[str],
    record_to_id: Dict[bytes, int],
    unk_id: int,
    indices: np.ndarray,
) -> np.ndarray:
    if not cond_keys:
        return np.zeros(len(indices), dtype=np.int64)
    arrays = load_condition_arrays(h5_file, split, cond_keys, indices)
    structured = np.rec.fromarrays(arrays)
    return np.array([record_to_id.get(record.tobytes(), unk_id) for record in structured], dtype=np.int64)


def load_vocab_json(vocab_path: Path) -> Dict[str, int]:
    vocab_data = json.loads(vocab_path.read_text())
    if isinstance(vocab_data, dict):
        if all(isinstance(v, int) for v in vocab_data.values()):
            return {str(k): int(v) for k, v in vocab_data.items()}
        if "stoi" in vocab_data and isinstance(vocab_data["stoi"], dict):
            return {str(k): int(v) for k, v in vocab_data["stoi"].items()}
    if isinstance(vocab_data, list):
        return {str(token): idx for idx, token in enumerate(vocab_data)}
    raise ValueError(f"Unsupported vocab format in {vocab_path}")


def ensure_vocab_tokens(token_to_id: Dict[str, int], genes: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray]:
    vocab = dict(token_to_id)
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab[token] = len(vocab)
    gene_ids = []
    for gene in genes:
        if gene not in vocab:
            vocab[gene] = len(vocab)
        gene_ids.append(vocab[gene])
    return vocab, np.asarray(gene_ids, dtype=np.int64)


def build_scratch_vocab(genes: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray]:
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    return ensure_vocab_tokens(vocab, genes)


def load_pretrained_assets(pretrained_dir: Path, genes: Sequence[str]) -> Tuple[Dict[str, int], np.ndarray, Dict[str, object], Dict[str, torch.Tensor]]:
    vocab = load_vocab_json(pretrained_dir / "vocab.json")
    vocab, gene_ids = ensure_vocab_tokens(vocab, genes)
    with open(pretrained_dir / "args.json", "r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    state = torch.load(pretrained_dir / "best_model.pt", map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Unsupported pretrained checkpoint format for scGPT baseline")
    return vocab, gene_ids, model_config, state


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


def load_transformer_model_class():
    spec = importlib.util.find_spec("scgpt")
    if spec is None or spec.origin is None:
        raise ImportError("scgpt is not installed in the selected environment")

    package_dir = Path(spec.origin).resolve().parent
    model_dir = package_dir / "model"
    model_file = model_dir / "model.py"
    if not model_file.exists():
        raise ImportError(f"Could not locate scGPT model module at {model_file}")

    if "scgpt" not in sys.modules:
        scgpt_pkg = types.ModuleType("scgpt")
        scgpt_pkg.__file__ = str(package_dir / "__init__.py")
        scgpt_pkg.__path__ = [str(package_dir)]
        sys.modules["scgpt"] = scgpt_pkg

    if "scgpt.model" not in sys.modules:
        scgpt_model_pkg = types.ModuleType("scgpt.model")
        scgpt_model_pkg.__file__ = str(model_dir / "__init__.py")
        scgpt_model_pkg.__path__ = [str(model_dir)]
        sys.modules["scgpt.model"] = scgpt_model_pkg

    module_name = "scgpt.model.model"
    module = sys.modules.get(module_name)
    if module is None:
        model_spec = importlib.util.spec_from_file_location(module_name, model_file)
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"Could not load import spec for {model_file}")
        module = importlib.util.module_from_spec(model_spec)
        sys.modules[module_name] = module
        model_spec.loader.exec_module(module)

    return module.TransformerModel


def build_model(
    args: argparse.Namespace,
    vocab: Dict[str, int],
    num_condition_labels: int,
    pretrained_config: Optional[Dict[str, object]] = None,
    pretrained_state: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.nn.Module:
    TransformerModel = load_transformer_model_class()

    model_kwargs = {
        "ntoken": len(vocab),
        "d_model": int(pretrained_config["embsize"]) if pretrained_config is not None else int(args.embsize),
        "nhead": int(pretrained_config["nheads"]) if pretrained_config is not None else int(args.nhead),
        "d_hid": int(pretrained_config["d_hid"]) if pretrained_config is not None else int(args.d_hid),
        "nlayers": int(pretrained_config["nlayers"]) if pretrained_config is not None else int(args.nlayers),
        "dropout": float(pretrained_config.get("dropout", args.dropout)) if pretrained_config is not None else float(args.dropout),
        "pad_token": PAD_TOKEN,
        "pad_value": PAD_VALUE,
        "do_mvc": False,
        "do_dab": False,
        "use_batch_labels": True,
        "num_batch_labels": int(num_condition_labels),
        "domain_spec_batchnorm": False,
        "input_emb_style": "continuous",
        "n_input_bins": 0,
        "cell_emb_style": "cls",
        "ecs_threshold": 0.0,
        "explicit_zero_prob": False,
        # Keep to vanilla PyTorch attention so flash-attn remains optional.
        "use_fast_transformer": False,
        "pre_norm": bool(pretrained_config.get("pre_norm", False)) if pretrained_config is not None else False,
        "n_cls": 1,
    }

    model = TransformerModel(
        model_kwargs["ntoken"],
        model_kwargs["d_model"],
        model_kwargs["nhead"],
        model_kwargs["d_hid"],
        model_kwargs["nlayers"],
        n_cls=model_kwargs["n_cls"],
        vocab=vocab,
        dropout=model_kwargs["dropout"],
        pad_token=model_kwargs["pad_token"],
        pad_value=model_kwargs["pad_value"],
        do_mvc=model_kwargs["do_mvc"],
        do_dab=model_kwargs["do_dab"],
        use_batch_labels=model_kwargs["use_batch_labels"],
        num_batch_labels=model_kwargs["num_batch_labels"],
        domain_spec_batchnorm=model_kwargs["domain_spec_batchnorm"],
        input_emb_style=model_kwargs["input_emb_style"],
        n_input_bins=model_kwargs["n_input_bins"],
        cell_emb_style=model_kwargs["cell_emb_style"],
        ecs_threshold=model_kwargs["ecs_threshold"],
        explicit_zero_prob=model_kwargs["explicit_zero_prob"],
        use_fast_transformer=model_kwargs["use_fast_transformer"],
        pre_norm=model_kwargs["pre_norm"],
    )

    if pretrained_state is not None:
        current_state = model.state_dict()
        for key, value in pretrained_state.items():
            if key not in current_state:
                continue
            if current_state[key].shape == value.shape:
                current_state[key] = value
            elif key == "encoder.embedding.weight" and current_state[key].ndim == 2 and value.ndim == 2 and current_state[key].shape[1] == value.shape[1]:
                rows = min(current_state[key].shape[0], value.shape[0])
                current_state[key][:rows] = value[:rows]
        model.load_state_dict(current_state)
    return model


def resolve_model_config(
    args: argparse.Namespace,
    pretrained_config: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "embsize": int(pretrained_config["embsize"]) if pretrained_config is not None else int(args.embsize),
        "nheads": int(pretrained_config["nheads"]) if pretrained_config is not None else int(args.nhead),
        "d_hid": int(pretrained_config["d_hid"]) if pretrained_config is not None else int(args.d_hid),
        "nlayers": int(pretrained_config["nlayers"]) if pretrained_config is not None else int(args.nlayers),
        "dropout": float(pretrained_config.get("dropout", args.dropout)) if pretrained_config is not None else float(args.dropout),
        "pre_norm": bool(pretrained_config.get("pre_norm", False)) if pretrained_config is not None else False,
    }


def initialize_model_for_training(
    *,
    args: argparse.Namespace,
    gene_names: Sequence[str],
    num_condition_labels: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, int], np.ndarray, Dict[str, object]]:
    if args.init_mode == "pretrained":
        print(f"Initializing model from pretrained checkpoint at {args.pretrained_dir}")
        vocab, gene_ids, pretrained_config, pretrained_state = load_pretrained_assets(Path(args.pretrained_dir), gene_names)
    else:
        vocab, gene_ids = build_scratch_vocab(gene_names)
        pretrained_config = None
        pretrained_state = None

    model = build_model(
        args=args,
        vocab=vocab,
        num_condition_labels=num_condition_labels,
        pretrained_config=pretrained_config,
        pretrained_state=pretrained_state,
    ).to(device)
    model_config = resolve_model_config(args, pretrained_config)
    return model, vocab, gene_ids, model_config


def initialize_model_from_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    gene_names: Sequence[str],
    num_condition_labels: int,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, int], np.ndarray, Dict[str, object]]:
    vocab, gene_ids, checkpoint_config, checkpoint_state = load_pretrained_assets(checkpoint_dir, gene_names)
    model = build_model(
        args=args,
        vocab=vocab,
        num_condition_labels=num_condition_labels,
        pretrained_config=checkpoint_config,
        pretrained_state=checkpoint_state,
    ).to(device)
    return model, vocab, gene_ids, checkpoint_config


def load_checkpoint_metadata(checkpoint_dir: Path) -> Dict[str, object]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_checkpoint_dir(
    *,
    checkpoint_dir: Path,
    model_state: Dict[str, torch.Tensor],
    vocab: Dict[str, int],
    model_config: Dict[str, object],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cpu_state = {key: value.detach().cpu() for key, value in model_state.items()}
    torch.save(cpu_state, checkpoint_dir / "best_model.pt")
    (checkpoint_dir / "args.json").write_text(json.dumps(model_config, indent=2, sort_keys=True), encoding="utf-8")
    (checkpoint_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True), encoding="utf-8")
    if metadata is not None:
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def load_numeric_batch(group: h5py.Group, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(indices)
    sorted_idx = indices[order]
    restore = np.argsort(order)
    counts = group["counts"][sorted_idx][restore].astype(np.float32, copy=False)
    missing = group["missingness_mask"][sorted_idx][restore].astype(bool, copy=False)
    return counts, missing


def sample_train_masks(valid_mask: np.ndarray, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
    mask = (rng.random(valid_mask.shape) < float(mask_ratio)) & valid_mask
    valid_counts = valid_mask.sum(axis=1)
    masked_counts = mask.sum(axis=1)

    none_masked = np.where((masked_counts == 0) & (valid_counts > 1))[0]
    for row in none_masked:
        candidates = np.flatnonzero(valid_mask[row])
        pick = int(rng.choice(candidates))
        mask[row, pick] = True

    all_masked = np.where(mask.sum(axis=1) >= valid_counts)[0]
    for row in all_masked:
        candidates = np.flatnonzero(valid_mask[row])
        if candidates.size == 0:
            continue
        keep = int(rng.choice(candidates))
        mask[row, keep] = False
    return mask


def prepare_model_batch(
    counts: np.ndarray,
    missingness_mask: np.ndarray,
    target_mask: np.ndarray,
    gene_ids: np.ndarray,
    cls_token_id: int,
    norm_target_sum: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    valid_mask = ~missingness_mask
    target_mask = target_mask & valid_mask

    retained = counts.copy()
    retained[target_mask] = 0.0
    retained[~valid_mask] = 0.0

    retained_library = retained.sum(axis=1, keepdims=True)
    retained_library = np.clip(retained_library, 1.0, None)

    normalized_input = np.log1p(norm_target_sum * retained / retained_library)
    normalized_target = np.log1p(norm_target_sum * np.clip(counts, 0.0, None) / retained_library)

    normalized_input[~valid_mask] = 0.0
    normalized_target[~valid_mask] = 0.0
    normalized_input[target_mask] = MASK_VALUE

    batch_size = counts.shape[0]
    seq_gene_ids = np.broadcast_to(
        np.concatenate(([cls_token_id], gene_ids)),
        (batch_size, gene_ids.shape[0] + 1),
    )
    seq_values = np.concatenate(
        (np.zeros((batch_size, 1), dtype=np.float32), normalized_input.astype(np.float32)),
        axis=1,
    )
    seq_targets = np.concatenate(
        (np.zeros((batch_size, 1), dtype=np.float32), normalized_target.astype(np.float32)),
        axis=1,
    )
    seq_loss_mask = np.concatenate(
        (np.zeros((batch_size, 1), dtype=bool), target_mask.astype(bool)),
        axis=1,
    )
    pad_mask = np.zeros_like(seq_loss_mask, dtype=bool)

    return {
        "gene_ids": torch.as_tensor(seq_gene_ids, dtype=torch.long, device=device),
        "values": torch.as_tensor(seq_values, dtype=torch.float32, device=device),
        "target_values": torch.as_tensor(seq_targets, dtype=torch.float32, device=device),
        "loss_mask": torch.as_tensor(seq_loss_mask, dtype=torch.bool, device=device),
        "src_key_padding_mask": torch.as_tensor(pad_mask, dtype=torch.bool, device=device),
        "retained_library": torch.as_tensor(retained_library, dtype=torch.float32, device=device),
        "valid_mask": torch.as_tensor(valid_mask, dtype=torch.bool, device=device),
        "target_mask": torch.as_tensor(target_mask, dtype=torch.bool, device=device),
        "raw_counts": torch.as_tensor(counts, dtype=torch.float32, device=device),
    }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(loss_mask):
        return torch.zeros((), device=prediction.device)
    return torch.mean((prediction[loss_mask] - target[loss_mask]) ** 2)


def extract_mlm_output(output_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    output = output_dict["mlm_output"]
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    return output


def run_epoch(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    group: h5py.Group,
    indices: np.ndarray,
    condition_ids: np.ndarray,
    batch_size: int,
    mask_ratio: float,
    gene_ids: np.ndarray,
    cls_token_id: int,
    norm_target_sum: float,
    device: torch.device,
    seed: int,
    progress_tracker: Optional[ProgressTracker] = None,
    phase_label: str = "",
    show_console_progress: bool = False,
) -> float:
    is_train = optimizer is not None
    rng = np.random.default_rng(seed)
    positions = np.arange(indices.shape[0], dtype=np.int64)
    if is_train:
        rng.shuffle(positions)
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    num_batches = count_batches(len(positions), batch_size)
    if progress_tracker is not None and num_batches == 0:
        progress_tracker.set_phase(phase_label or ("train" if is_train else "val"))
    iterator = range(0, len(positions), batch_size)
    progress_bar = None
    if show_console_progress and num_batches > 0:
        progress_bar = tqdm(
            total=num_batches,
            desc=phase_label or ("train" if is_train else "val"),
            leave=False,
            dynamic_ncols=True,
        )

    for batch_num, start in enumerate(iterator, start=1):
        batch_pos = positions[start:start + batch_size]
        batch_indices = indices[batch_pos]
        counts, missingness_mask = load_numeric_batch(group, batch_indices)
        train_mask = sample_train_masks(~missingness_mask, mask_ratio, rng)
        batch = prepare_model_batch(
            counts=counts,
            missingness_mask=missingness_mask,
            target_mask=train_mask,
            gene_ids=gene_ids,
            cls_token_id=cls_token_id,
            norm_target_sum=norm_target_sum,
            device=device,
        )
        batch_condition_ids = torch.as_tensor(condition_ids[batch_pos], dtype=torch.long, device=device)

        with torch.set_grad_enabled(is_train):
            output = model(
                batch["gene_ids"],
                batch["values"],
                src_key_padding_mask=batch["src_key_padding_mask"],
                batch_labels=batch_condition_ids,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
            )
            loss = masked_mse(extract_mlm_output(output), batch["target_values"], batch["loss_mask"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.item()) * len(batch_pos)
        total_examples += len(batch_pos)
        if progress_tracker is not None:
            progress_tracker.advance(f"{phase_label} [{batch_num}/{num_batches}]")
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
    if progress_bar is not None:
        progress_bar.close()
    return total_loss / max(total_examples, 1)


def train_model(
    *,
    model: torch.nn.Module,
    train_group: h5py.Group,
    val_group: h5py.Group,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_condition_ids: np.ndarray,
    val_condition_ids: np.ndarray,
    batch_size: int,
    eval_batch_size: int,
    epochs: int,
    lr: float,
    mask_ratio: float,
    gene_ids: np.ndarray,
    cls_token_id: int,
    norm_target_sum: float,
    device: torch.device,
    seed: int,
    progress_tracker: Optional[ProgressTracker] = None,
    show_console_progress: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf

    if epochs > 0:
        for epoch in range(epochs):
            train_loss = run_epoch(
                model=model,
                optimizer=optimizer,
                group=train_group,
                indices=train_indices,
                condition_ids=train_condition_ids,
                batch_size=batch_size,
                mask_ratio=mask_ratio,
                gene_ids=gene_ids,
                cls_token_id=cls_token_id,
                norm_target_sum=norm_target_sum,
                device=device,
                seed=seed + epoch,
                progress_tracker=progress_tracker,
                phase_label=f"train {epoch + 1}/{epochs}",
                show_console_progress=show_console_progress,
            )
            val_loss = run_epoch(
                model=model,
                optimizer=None,
                group=val_group,
                indices=val_indices,
                condition_ids=val_condition_ids,
                batch_size=eval_batch_size,
                mask_ratio=mask_ratio,
                gene_ids=gene_ids,
                cls_token_id=cls_token_id,
                norm_target_sum=norm_target_sum,
                device=device,
                seed=seed + 10_000,
                progress_tracker=progress_tracker,
                phase_label=f"val {epoch + 1}/{epochs}",
                show_console_progress=show_console_progress,
            )
            print(
                f"[scGPT] epoch {epoch + 1}/{epochs} "
                f"| train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_state, best_val


@torch.no_grad()
def impute_test_split(
    *,
    model: torch.nn.Module,
    group: h5py.Group,
    indices: np.ndarray,
    condition_ids: np.ndarray,
    batch_size: int,
    target_mask_full: np.ndarray,
    gene_ids: np.ndarray,
    cls_token_id: int,
    norm_target_sum: float,
    device: torch.device,
    progress_tracker: Optional[ProgressTracker] = None,
    show_console_progress: bool = False,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    num_batches = count_batches(len(indices), batch_size)
    if progress_tracker is not None and num_batches == 0:
        progress_tracker.set_phase("test")
    progress_bar = None
    if show_console_progress and num_batches > 0:
        progress_bar = tqdm(
            total=num_batches,
            desc="test imputation",
            leave=True,
            dynamic_ncols=True,
        )

    for batch_num, start in enumerate(range(0, len(indices), batch_size), start=1):
        batch_indices = indices[start:start + batch_size]
        counts, missingness_mask = load_numeric_batch(group, batch_indices)
        batch_target_mask = target_mask_full[batch_indices]
        batch = prepare_model_batch(
            counts=counts,
            missingness_mask=missingness_mask,
            target_mask=batch_target_mask,
            gene_ids=gene_ids,
            cls_token_id=cls_token_id,
            norm_target_sum=norm_target_sum,
            device=device,
        )
        batch_condition_ids = torch.as_tensor(condition_ids[start:start + len(batch_indices)], dtype=torch.long, device=device)
        output = model(
            batch["gene_ids"],
            batch["values"],
            src_key_padding_mask=batch["src_key_padding_mask"],
            batch_labels=batch_condition_ids,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
        )
        predicted_norm = extract_mlm_output(output)[:, 1:]
        predicted_counts = torch.expm1(predicted_norm).clamp(min=0.0)
        predicted_counts = predicted_counts * (batch["retained_library"] / float(norm_target_sum))
        imputed = batch["raw_counts"].clone()
        imputed[batch["target_mask"]] = predicted_counts[batch["target_mask"]]
        outputs.append(imputed.detach().cpu().numpy().astype(np.float32))
        if progress_tracker is not None:
            progress_tracker.advance(f"test [{batch_num}/{num_batches}]")
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()
    return np.concatenate(outputs, axis=0)


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
        test_group = h5_file["test"]

        train_indices = choose_indices(train_group["counts"].shape[0], args.max_train_cells, args.seed + 11)
        val_indices = choose_indices(val_group["counts"].shape[0], args.max_val_cells, args.seed + 17)
        test_indices = choose_indices(test_group["counts"].shape[0], args.max_test_cells, args.seed + 23)
        total_progress_units = (
            args.epochs * count_batches(len(train_indices), args.batch_size)
            + args.epochs * count_batches(len(val_indices), args.eval_batch_size)
            + count_batches(len(test_indices), args.eval_batch_size)
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
        test_condition_ids = map_condition_ids(
            h5_file,
            "test",
            args.cond_keys,
            record_to_id,
            num_condition_labels - 1,
            test_indices,
        )

        model, vocab, gene_ids, model_config = initialize_model_for_training(
            args=args,
            gene_names=gene_names,
            num_condition_labels=num_condition_labels,
            device=device,
        )
        progress_tracker.set_phase("initialized model", force=True)

        cls_token_id = vocab[CLS_TOKEN]
        model, best_state, _ = train_model(
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

        if args.save_checkpoint_dir is not None:
            save_checkpoint_dir(
                checkpoint_dir=Path(args.save_checkpoint_dir),
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
                },
            )
            print(f"[scGPT] saved checkpoint to {args.save_checkpoint_dir}")

        test_target_mask = np.load(args.mask_file).astype(bool)
        imputed = impute_test_split(
            model=model,
            group=test_group,
            indices=test_indices,
            condition_ids=test_condition_ids,
            batch_size=args.eval_batch_size,
            target_mask_full=test_target_mask,
            gene_ids=gene_ids,
            cls_token_id=cls_token_id,
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
