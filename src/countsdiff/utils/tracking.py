"""
Shared Weights & Biases tracking and run-resolution utilities.
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import wandb
import yaml


WANDB_ENTITY = "anonymous2-icml"
WANDB_PROJECT = "countsdiff-icml"
WANDB_PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
WANDB_WRITE_KEY_ENV = "WANDB_API_KEY"
_CONFIG_NAMESPACE_PREFIX = "config_"
_CONFIG_SECTIONS = {"model", "data", "scheduler", "training", "generation", "tracking"}


class RunResolutionError(RuntimeError):
    """Raised when a run reference cannot be resolved uniquely."""


def normalize_config_literals(obj: Any) -> Any:
    """
    Recursively convert stringified Python literals into real Python objects.
    """
    if isinstance(obj, dict):
        return {key: normalize_config_literals(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [normalize_config_literals(value) for value in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return obj
        return normalize_config_literals(parsed)
    return obj


def make_wandb_safe(obj: Any) -> Any:
    """
    Convert nested config/metric values into W&B-friendly JSON-serializable objects.
    """
    if isinstance(obj, dict):
        return {str(key): make_wandb_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_wandb_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def _extract_legacy_id(run_ref: str) -> str:
    return run_ref.strip().split("/")[-1]


def _try_direct_run(api: wandb.Api, run_ref: str) -> Optional[Any]:
    candidate_refs = []
    normalized = run_ref.strip()
    if not normalized:
        return None

    if normalized.count("/") == 2:
        candidate_refs.append(normalized)
    elif normalized.count("/") == 1:
        candidate_refs.append(f"{WANDB_ENTITY}/{normalized}")
    else:
        candidate_refs.append(f"{WANDB_PROJECT_PATH}/{normalized}")

    for candidate in candidate_refs:
        try:
            return api.run(candidate)
        except Exception:
            continue
    return None


def _unflatten_exported_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for key, value in raw_config.items():
        if not key.startswith(_CONFIG_NAMESPACE_PREFIX):
            continue

        remainder = key[len(_CONFIG_NAMESPACE_PREFIX):]
        head, sep, tail = remainder.partition("_")
        if sep and head in _CONFIG_SECTIONS:
            section = config.setdefault(head, {})
            section[tail] = value
        else:
            config[remainder] = value

    return normalize_config_literals(config)


def _has_useful_run_config(raw_config: Dict[str, Any]) -> bool:
    return any(key.startswith(_CONFIG_NAMESPACE_PREFIX) for key in raw_config) or any(
        key in raw_config for key in _CONFIG_SECTIONS.union({"sys_id", "sys_name"})
    )


def _load_config_from_run_file(run: Any) -> Dict[str, Any]:
    try:
        config_file = run.file("config.yaml")
        if config_file is None:
            return {}
        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = config_file.download(root=tmpdir, replace=True)
            with open(downloaded.name, "r") as handle:
                raw_data = yaml.safe_load(handle) or {}
    except Exception:
        return {}

    normalized = {}
    for key, value in raw_data.items():
        if isinstance(value, dict) and "value" in value:
            normalized[key] = value["value"]
        else:
            normalized[key] = value
    return normalized


def _extract_logged_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    if any(key.startswith(_CONFIG_NAMESPACE_PREFIX) for key in raw_config):
        return _unflatten_exported_config(raw_config)

    known_config = {
        key: value
        for key, value in raw_config.items()
        if key in _CONFIG_SECTIONS or key in {"run_name", "checkpoint_subdir"}
    }
    if known_config:
        return normalize_config_literals(copy.deepcopy(known_config))

    return normalize_config_literals(copy.deepcopy(raw_config))


def _resolve_run_name(run: Any, config: Dict[str, Any], raw_config: Dict[str, Any]) -> str:
    tracking_config = config.get("tracking", {}) if isinstance(config.get("tracking"), dict) else {}
    for candidate in (
        config.get("run_name"),
        tracking_config.get("run_name"),
        raw_config.get("config_run_name"),
        raw_config.get("sys_name"),
        run.name,
    ):
        if candidate not in (None, "", "None"):
            return str(candidate)
    return str(run.name)


def _resolve_checkpoint_subdir(config: Dict[str, Any], run_name: str) -> str:
    tracking_config = config.get("tracking", {}) if isinstance(config.get("tracking"), dict) else {}
    for candidate in (
        config.get("checkpoint_subdir"),
        tracking_config.get("checkpoint_subdir"),
        run_name,
    ):
        if candidate not in (None, "", "None"):
            return str(candidate)
    return str(run_name)


def _resolve_legacy_run_id(run: Any, raw_config: Dict[str, Any]) -> Optional[str]:
    for candidate in (
        raw_config.get("sys_id"),
        raw_config.get("legacy_run_id"),
        raw_config.get("tracking_legacy_run_id"),
    ):
        if candidate not in (None, "", "None"):
            return str(candidate)
    return None


@dataclass
class ResolvedRun:
    """
    Fully resolved run metadata for loading or resuming an experiment.
    """

    run_id: str
    run_path: str
    display_name: str
    legacy_run_id: Optional[str]
    run_name: str
    checkpoint_subdir: str
    config: Dict[str, Any]
    raw_config: Dict[str, Any]


def resolved_config_for_run(run: Any) -> ResolvedRun:
    raw_config = dict(run.config)
    if not _has_useful_run_config(raw_config):
        file_config = _load_config_from_run_file(run)
        if file_config:
            raw_config = file_config
    config = _extract_logged_config(raw_config)
    run_name = _resolve_run_name(run, config, raw_config)
    checkpoint_subdir = _resolve_checkpoint_subdir(config, run_name)
    legacy_run_id = _resolve_legacy_run_id(run, raw_config)

    config["run_name"] = run_name
    config["checkpoint_subdir"] = checkpoint_subdir
    tracking_config = config.get("tracking", {}) if isinstance(config.get("tracking"), dict) else {}
    tracking_config.update({
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "wandb_run_id": run.id,
        "wandb_run_path": "/".join(run.path),
        "display_name": run.name,
        "run_name": run_name,
        "checkpoint_subdir": checkpoint_subdir,
    })
    if legacy_run_id:
        tracking_config["legacy_run_id"] = legacy_run_id
    config["tracking"] = tracking_config

    return ResolvedRun(
        run_id=str(run.id),
        run_path="/".join(run.path),
        display_name=str(run.name),
        legacy_run_id=legacy_run_id,
        run_name=run_name,
        checkpoint_subdir=checkpoint_subdir,
        config=config,
        raw_config=raw_config,
    )


def resolve_run_reference(run_ref: str, api: Optional[wandb.Api] = None) -> ResolvedRun:
    """
    Resolve a user-supplied run reference against the public W&B project.

    Resolution order:
    1. Direct W&B run path or W&B-native run ID.
    2. Exported legacy Neptune ID preserved in run metadata.
    3. W&B display name fallback.
    """
    api = api or wandb.Api()

    direct_run = _try_direct_run(api, run_ref)
    if direct_run is not None:
        return resolved_config_for_run(direct_run)

    legacy_id = _extract_legacy_id(run_ref)
    metadata_matches = []
    name_matches = []

    for run in api.runs(WANDB_PROJECT_PATH):
        raw_config = dict(run.config)
        if raw_config.get("sys_id") == legacy_id:
            metadata_matches.append(run)
            continue
        if raw_config.get("legacy_run_id") == legacy_id:
            metadata_matches.append(run)
            continue
        if run.name == legacy_id:
            name_matches.append(run)

    if len(metadata_matches) == 1:
        return resolved_config_for_run(metadata_matches[0])
    if len(metadata_matches) > 1:
        matches = ", ".join(sorted(run.id for run in metadata_matches))
        raise RunResolutionError(
            f"Run reference '{run_ref}' matched multiple exported runs by legacy metadata: {matches}"
        )

    if len(name_matches) == 1:
        return resolved_config_for_run(name_matches[0])
    if len(name_matches) > 1:
        matches = ", ".join(sorted(run.id for run in name_matches))
        raise RunResolutionError(
            f"Run reference '{run_ref}' matched multiple runs by W&B display name: {matches}"
        )

    raise RunResolutionError(
        f"Could not resolve run reference '{run_ref}' in public W&B project {WANDB_PROJECT_PATH}"
    )


def build_logged_config(
    config: Dict[str, Any],
    *,
    run_name: str,
    checkpoint_subdir: str,
    wandb_run_id: Optional[str] = None,
    wandb_run_path: Optional[str] = None,
    legacy_run_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create the canonical config payload stored on new or resumed W&B runs.
    """
    logged_config = normalize_config_literals(make_wandb_safe(copy.deepcopy(config)))
    logged_config["run_name"] = run_name
    logged_config["checkpoint_subdir"] = checkpoint_subdir

    tracking_config = logged_config.get("tracking", {}) if isinstance(logged_config.get("tracking"), dict) else {}
    tracking_config.update({
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "run_name": run_name,
        "checkpoint_subdir": checkpoint_subdir,
    })
    if wandb_run_id:
        tracking_config["wandb_run_id"] = wandb_run_id
    if wandb_run_path:
        tracking_config["wandb_run_path"] = wandb_run_path
    if legacy_run_id:
        tracking_config["legacy_run_id"] = legacy_run_id
    if display_name:
        tracking_config["display_name"] = display_name
    logged_config["tracking"] = tracking_config
    return logged_config


def require_write_api_key() -> str:
    api_key = os.environ.get(WANDB_WRITE_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"{WANDB_WRITE_KEY_ENV} must be set to create or resume tracked W&B runs."
        )
    return api_key


class WandbTracker:
    """Thin wrapper over a W&B run for training-time config and metric logging."""

    def __init__(self, run: Any):
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def update_config(self, config: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.run.config.update(make_wandb_safe(config), allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        self.run.log(make_wandb_safe(metrics), step=step)

    def finish(self) -> None:
        if not self.enabled:
            return
        self.run.finish()


def init_wandb_tracker(
    *,
    logged_config: Dict[str, Any],
    tags: Optional[Iterable[str]] = None,
    resume_run_id: Optional[str] = None,
    display_name: Optional[str] = None,
) -> WandbTracker:
    """
    Initialize a W&B run for writing metrics/configs.
    """
    api_key = require_write_api_key()
    wandb.login(key=api_key, relogin=True)

    init_kwargs: Dict[str, Any] = {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "config": make_wandb_safe(logged_config),
    }
    if tags:
        init_kwargs["tags"] = list(tags)
    if resume_run_id is not None:
        init_kwargs["id"] = resume_run_id
        init_kwargs["resume"] = "must"
    elif display_name:
        init_kwargs["name"] = display_name

    run = wandb.init(**init_kwargs)
    return WandbTracker(run)
