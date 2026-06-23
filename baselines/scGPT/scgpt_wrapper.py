from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np


class SCGPTWrapper:
    """Thin wrapper that runs the scGPT baseline in a dedicated conda env."""

    def __init__(
        self,
        *,
        conda_env: str = "scgpt-baseline",
        work_dir: Optional[str] = None,
        runner_path: Optional[str] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.conda_env = conda_env
        self.work_dir = Path(work_dir) if work_dir is not None else self.repo_root / "data" / "dnadiff" / "scgpt_runs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.runner_path = Path(runner_path) if runner_path is not None else Path(__file__).resolve().parent / "run_scgpt_imputation.py"
        self.train_path = Path(__file__).resolve().parent / "train_scgpt.py"
        self.impute_path = Path(__file__).resolve().parent / "impute_scgpt.py"

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
            prefix="scgpt_progress_",
            suffix=".json",
            dir=self.work_dir,
        )
        os.close(progress_fd)
        progress_path = Path(progress_path_str)
        if progress_path.exists():
            progress_path.unlink()

        process = subprocess.Popen(
            [*cmd, "--progress-path", str(progress_path)],
            cwd=self.repo_root,
        )
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
        cond_keys: Iterable[str],
        device: str,
        init_mode: str,
        seed: int,
        epochs: int,
        batch_size: int,
        eval_batch_size: int,
        lr: float,
        train_mask_ratio: float,
        norm_target_sum: float,
        embsize: int,
        d_hid: int,
        nhead: int,
        nlayers: int,
        dropout: float,
        max_train_cells: Optional[int] = None,
        max_val_cells: Optional[int] = None,
        pretrained_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> str:
        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.train_path),
            "--data-file",
            data_file,
            "--checkpoint-dir",
            checkpoint_dir,
            "--device",
            device,
            "--init-mode",
            init_mode,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(eval_batch_size),
            "--lr",
            str(lr),
            "--train-mask-ratio",
            str(train_mask_ratio),
            "--norm-target-sum",
            str(norm_target_sum),
            "--embsize",
            str(embsize),
            "--d-hid",
            str(d_hid),
            "--nhead",
            str(nhead),
            "--nlayers",
            str(nlayers),
            "--dropout",
            str(dropout),
        ]
        if pretrained_dir is not None:
            cmd.extend(["--pretrained-dir", pretrained_dir])
        if run_name is not None:
            cmd.extend(["--run-name", run_name])
        for key in cond_keys:
            cmd.extend(["--cond-key", key])
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
        cond_keys: Iterable[str],
        device: str,
        seed: int,
        eval_batch_size: int,
        norm_target_sum: float,
        max_test_cells: Optional[int] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> np.ndarray:
        fd, output_path = tempfile.mkstemp(
            prefix="scgpt_imputation_",
            suffix=".npy",
            dir=self.work_dir,
        )
        os.close(fd)

        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.impute_path),
            "--data-file",
            data_file,
            "--mask-file",
            mask_file,
            "--checkpoint-dir",
            checkpoint_dir,
            "--output-path",
            output_path,
            "--device",
            device,
            "--seed",
            str(seed),
            "--eval-batch-size",
            str(eval_batch_size),
            "--norm-target-sum",
            str(norm_target_sum),
        ]
        for key in cond_keys:
            cmd.extend(["--cond-key", key])
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
        cond_keys: Iterable[str],
        device: str,
        init_mode: str,
        seed: int,
        epochs: int,
        batch_size: int,
        eval_batch_size: int,
        lr: float,
        train_mask_ratio: float,
        norm_target_sum: float,
        embsize: int,
        d_hid: int,
        nhead: int,
        nlayers: int,
        dropout: float,
        max_train_cells: Optional[int] = None,
        max_val_cells: Optional[int] = None,
        max_test_cells: Optional[int] = None,
        pretrained_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
        save_checkpoint_dir: Optional[str] = None,
    ) -> np.ndarray:
        fd, output_path = tempfile.mkstemp(
            prefix="scgpt_imputation_",
            suffix=".npy",
            dir=self.work_dir,
        )
        os.close(fd)

        cmd = [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.runner_path),
            "--data-file",
            data_file,
            "--mask-file",
            mask_file,
            "--output-path",
            output_path,
            "--device",
            device,
            "--init-mode",
            init_mode,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--eval-batch-size",
            str(eval_batch_size),
            "--lr",
            str(lr),
            "--train-mask-ratio",
            str(train_mask_ratio),
            "--norm-target-sum",
            str(norm_target_sum),
            "--embsize",
            str(embsize),
            "--d-hid",
            str(d_hid),
            "--nhead",
            str(nhead),
            "--nlayers",
            str(nlayers),
            "--dropout",
            str(dropout),
        ]
        if pretrained_dir is not None:
            cmd.extend(["--pretrained-dir", pretrained_dir])
        if run_name is not None:
            cmd.extend(["--run-name", run_name])
        if save_checkpoint_dir is not None:
            cmd.extend(["--save-checkpoint-dir", save_checkpoint_dir])
        for key in cond_keys:
            cmd.extend(["--cond-key", key])
        if max_train_cells is not None:
            cmd.extend(["--max-train-cells", str(max_train_cells)])
        if max_val_cells is not None:
            cmd.extend(["--max-val-cells", str(max_val_cells)])
        if max_test_cells is not None:
            cmd.extend(["--max-test-cells", str(max_test_cells)])

        self._run_process(cmd, progress_callback=progress_callback)
        try:
            return np.load(output_path)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
