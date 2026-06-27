from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


class XTrimoGeneWrapper:
    """Thin wrapper that runs the xTrimoGene baseline in a dedicated conda env."""

    def __init__(
        self,
        *,
        conda_env: str = "xtrimogene-baseline",
        work_dir: Optional[str] = None,
        train_path: Optional[str] = None,
        impute_path: Optional[str] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.conda_env = conda_env
        self.work_dir = Path(work_dir) if work_dir is not None else self.repo_root / "data" / "dnadiff" / "xtrimogene_runs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = Path(train_path) if train_path is not None else Path(__file__).resolve().parent / "train_xtrimogene.py"
        self.impute_path = Path(impute_path) if impute_path is not None else Path(__file__).resolve().parent / "impute_xtrimogene.py"

    @staticmethod
    def _load_progress(progress_path: Path) -> tuple[Optional[float], Optional[str]]:
        if not progress_path.exists():
            return None, None
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None, None
        fraction = payload.get("fraction")
        phase = payload.get("phase")
        if not isinstance(fraction, (int, float)):
            return None, None
        return float(fraction), str(phase) if phase is not None else None

    def _run_process(
        self,
        cmd: list[str],
        *,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> None:
        progress_fd, progress_path_str = tempfile.mkstemp(
            prefix="xtrimogene_progress_",
            suffix=".json",
            dir=self.work_dir,
        )
        os.close(progress_fd)
        progress_path = Path(progress_path_str)
        if progress_path.exists():
            progress_path.unlink()

        process = subprocess.Popen([*cmd, "--progress-path", str(progress_path)], cwd=self.repo_root)
        last_fraction: Optional[float] = None
        last_phase: Optional[str] = None
        if progress_callback is not None:
            progress_callback(0.0, "launching")

        try:
            while True:
                return_code = process.poll()
                fraction, phase = self._load_progress(progress_path)
                if (
                    progress_callback is not None
                    and fraction is not None
                    and (last_fraction is None or abs(fraction - last_fraction) > 1e-6 or phase != last_phase)
                ):
                    progress_callback(fraction, phase)
                    last_fraction = fraction
                    last_phase = phase

                if return_code is not None:
                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, cmd)
                    break
                time.sleep(0.5)

            fraction, phase = self._load_progress(progress_path)
            if progress_callback is not None:
                if fraction is not None:
                    progress_callback(fraction, phase)
                progress_callback(1.0, "saved")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if progress_path.exists():
                progress_path.unlink()

    def train_model(
        self,
        *,
        data_file: str,
        checkpoint_dir: str,
        device: str,
        seed: int,
        epochs: int,
        batch_size: int,
        eval_batch_size: int,
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
        max_train_cells: Optional[int] = None,
        max_val_cells: Optional[int] = None,
        run_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> str:
        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", str(self.train_path),
            "--data-file", data_file,
            "--checkpoint-dir", checkpoint_dir,
            "--device", device,
            "--seed", str(seed),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--eval-batch-size", str(eval_batch_size),
            "--grad-accum-steps", str(grad_accum_steps),
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--warmup-steps", str(warmup_steps),
            "--value-mask-prob", str(value_mask_prob),
            "--zero-mask-prob", str(zero_mask_prob),
            "--replace-prob", str(replace_prob),
            "--random-token-prob", str(random_token_prob),
            "--norm-target-sum", str(norm_target_sum),
            "--gradient-clip-val", str(gradient_clip_val),
        ]
        if run_name is not None:
            cmd.extend(["--run-name", run_name])
        if max_train_cells is not None:
            cmd.extend(["--max-train-cells", str(max_train_cells)])
        if max_val_cells is not None:
            cmd.extend(["--max-val-cells", str(max_val_cells)])

        self._run_process(cmd, progress_callback=progress_callback)
        return checkpoint_dir

    def impute_from_checkpoint(
        self,
        *,
        data_file: str,
        mask_file: str,
        checkpoint_dir: str,
        device: str,
        seed: int,
        eval_batch_size: int,
        norm_target_sum: float,
        max_test_cells: Optional[int] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> np.ndarray:
        fd, output_path = tempfile.mkstemp(prefix="xtrimogene_imputation_", suffix=".npy", dir=self.work_dir)
        os.close(fd)

        cmd = [
            "conda", "run", "-n", self.conda_env,
            "python", str(self.impute_path),
            "--data-file", data_file,
            "--mask-file", mask_file,
            "--checkpoint-dir", checkpoint_dir,
            "--output-path", output_path,
            "--device", device,
            "--seed", str(seed),
            "--eval-batch-size", str(eval_batch_size),
            "--norm-target-sum", str(norm_target_sum),
        ]
        if max_test_cells is not None:
            cmd.extend(["--max-test-cells", str(max_test_cells)])

        self._run_process(cmd, progress_callback=progress_callback)
        try:
            return np.load(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def impute_data(
        self,
        *,
        data_file: str,
        mask_file: str,
        device: str,
        seed: int,
        epochs: int,
        batch_size: int,
        eval_batch_size: int,
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
        max_train_cells: Optional[int] = None,
        max_val_cells: Optional[int] = None,
        max_test_cells: Optional[int] = None,
        run_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
        save_checkpoint_dir: Optional[str] = None,
        cond_keys: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        del cond_keys
        checkpoint_dir = save_checkpoint_dir
        cleanup_checkpoint_dir = False
        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp(prefix="xtrimogene_ckpt_", dir=self.work_dir)
            cleanup_checkpoint_dir = True

        try:
            def train_progress(fraction: float, phase: Optional[str]) -> None:
                if progress_callback is not None:
                    progress_callback(0.9 * float(fraction), phase)

            def impute_progress(fraction: float, phase: Optional[str]) -> None:
                if progress_callback is not None:
                    progress_callback(0.9 + 0.1 * float(fraction), phase)

            self.train_model(
                data_file=data_file,
                checkpoint_dir=checkpoint_dir,
                device=device,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                grad_accum_steps=grad_accum_steps,
                lr=lr,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps,
                value_mask_prob=value_mask_prob,
                zero_mask_prob=zero_mask_prob,
                replace_prob=replace_prob,
                random_token_prob=random_token_prob,
                norm_target_sum=norm_target_sum,
                gradient_clip_val=gradient_clip_val,
                max_train_cells=max_train_cells,
                max_val_cells=max_val_cells,
                run_name=run_name,
                progress_callback=train_progress,
            )
            return self.impute_from_checkpoint(
                data_file=data_file,
                mask_file=mask_file,
                checkpoint_dir=checkpoint_dir,
                device=device,
                seed=seed,
                eval_batch_size=eval_batch_size,
                norm_target_sum=norm_target_sum,
                max_test_cells=max_test_cells,
                progress_callback=impute_progress,
            )
        finally:
            if cleanup_checkpoint_dir and os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
