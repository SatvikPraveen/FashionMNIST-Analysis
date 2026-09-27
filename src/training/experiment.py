"""
Experiment tracking for FashionMNIST-Analysis.

Every training run gets a *run directory* holding:

* ``run.json``      – metadata: git commit, host, SLURM ids, device, torch
                      version, the full config snapshot, CLI argv, timings
                      and the final summary metrics.
* ``metrics.jsonl`` – one JSON object per logged step (epoch), so a sweep's
                      results can be aggregated with a few lines of pandas
                      (see ``src/cli/aggregate.py``).

MLflow and Weights & Biases are optional mirrors of the same information;
they are used only when enabled in the config *and* importable, so the
pipeline never hard-depends on them.

Usage:
    from src.training.experiment import ExperimentLogger, collect_run_metadata

    logger = ExperimentLogger(run_dir, run_name="tinyvgg_seed42", config=config)
    logger.log_params({"lr": 1e-3})
    logger.log_metrics({"val_acc": 0.93}, step=epoch)
    logger.finish({"test_acc": 0.931})
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> Dict[str, Any]:
    """Commit hash, branch and whether the tree had uncommitted changes."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "commit": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def slurm_info() -> Dict[str, Any]:
    """SLURM identifiers when running under a scheduler, else empty."""
    keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
            "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_NODELIST",
            "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE")
    return {k.lower(): os.environ[k] for k in keys if k in os.environ}


def device_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {"type": device.type}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update(name=props.name, total_memory_gb=round(props.total_memory / 1e9, 2),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    elif device.type == "mps":
        info["name"] = "Apple MPS"
    else:
        info["name"] = platform.processor() or "cpu"
    return info


def collect_run_metadata(config: Any, device: torch.device,
                         argv: Optional[list] = None) -> Dict[str, Any]:
    """Everything needed to reproduce / audit a run, as plain JSON."""
    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "argv": list(argv if argv is not None else sys.argv),
        "git": git_info(),
        "slurm": slurm_info(),
        "device": device_info(device),
        "config": cfg,
    }


# --------------------------------------------------------------------------- #
# Logger
# --------------------------------------------------------------------------- #

class ExperimentLogger:
    """
    File-first experiment logger with optional MLflow / W&B mirrors.

    Args:
        run_dir: Directory for ``run.json`` and ``metrics.jsonl``.
        run_name: Human-readable run name (also the MLflow/W&B run name).
        config: Project ``Config`` (or dict) snapshotted into ``run.json``.
        device: Device used for training (recorded in metadata).
        use_mlflow / use_wandb: Enable the mirrors. Silently disabled when
            the package is missing.
        mlflow_tracking_uri, experiment_name: MLflow settings.
        wandb_project: W&B project name.
        argv: CLI arguments to record (defaults to ``sys.argv``).
    """

    def __init__(self,
                 run_dir: str,
                 run_name: str,
                 config: Any = None,
                 device: Optional[torch.device] = None,
                 use_mlflow: bool = False,
                 use_wandb: bool = False,
                 mlflow_tracking_uri: Optional[str] = None,
                 experiment_name: str = "fashion-mnist",
                 wandb_project: str = "fashion-mnist",
                 argv: Optional[list] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._t0 = time.time()
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._run_path = self.run_dir / "run.json"
        self._params: Dict[str, Any] = {}
        self._finished = False

        self.meta = collect_run_metadata(config, device or torch.device("cpu"), argv)
        self.meta["run_name"] = run_name
        self._write_run_json()

        # Optional mirrors ------------------------------------------------- #
        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow  # type: ignore
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_name)
                mlflow.start_run(run_name=run_name)
                mlflow.set_tags({"git_commit": self.meta["git"].get("commit") or "",
                                 **{f"slurm_{k}": v for k, v in self.meta["slurm"].items()}})
                self._mlflow = mlflow
                log.info(f"MLflow tracking enabled (experiment='{experiment_name}')")
            except Exception as e:  # ImportError or a tracking-server error
                log.warning(f"MLflow requested but unavailable: {e}")

        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb = wandb.init(project=wandb_project, name=run_name,
                                         config=self.meta["config"], reinit=True)
                log.info(f"W&B tracking enabled (project='{wandb_project}')")
            except Exception as e:
                log.warning(f"W&B requested but unavailable: {e}")

    # ------------------------------------------------------------------ #
    def log_params(self, params: Dict[str, Any]) -> None:
        self._params.update(params)
        self.meta["params"] = self._params
        self._write_run_json()
        if self._mlflow:
            try:
                self._mlflow.log_params({k: str(v) for k, v in params.items()})
            except Exception as e:
                log.warning(f"mlflow.log_params failed: {e}")
        if self._wandb:
            self._wandb.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        record = {"step": step, "time": round(time.time() - self._t0, 3), **metrics}
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._mlflow:
            try:
                self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                          if isinstance(v, (int, float))}, step=step)
            except Exception as e:
                log.warning(f"mlflow.log_metrics failed: {e}")
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if self._mlflow and os.path.exists(path):
            try:
                self._mlflow.log_artifact(path)
            except Exception as e:
                log.warning(f"mlflow.log_artifact failed: {e}")
        if self._wandb and os.path.exists(path):
            self._wandb.save(path, policy="now")

    def finish(self, summary: Optional[Dict[str, Any]] = None, status: str = "finished") -> None:
        if self._finished:
            return
        self._finished = True
        self.meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.meta["duration_sec"] = round(time.time() - self._t0, 3)
        self.meta["status"] = status
        if summary:
            self.meta["summary"] = summary
        self._write_run_json()
        if self._mlflow:
            try:
                if summary:
                    self._mlflow.log_metrics({k: float(v) for k, v in summary.items()
                                              if isinstance(v, (int, float))})
                self._mlflow.end_run()
            except Exception as e:
                log.warning(f"mlflow finish failed: {e}")
        if self._wandb:
            if summary:
                for k, v in summary.items():
                    self._wandb.summary[k] = v
            self._wandb.finish()

    # ------------------------------------------------------------------ #
    def _write_run_json(self) -> None:
        tmp = self._run_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2, default=str)
        os.replace(tmp, self._run_path)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(status="failed" if exc_type else "finished")


def read_metrics(run_dir: str) -> list:
    """Load ``metrics.jsonl`` from a run directory as a list of dicts."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_run(run_dir: str) -> Dict[str, Any]:
    """Load ``run.json`` from a run directory."""
    with open(Path(run_dir) / "run.json") as f:
        return json.load(f)
