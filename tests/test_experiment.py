"""
Tests for experiment tracking, checkpoint resume and the AMP code path.
"""

import json

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import Config
from src.training.experiment import (
    ExperimentLogger, collect_run_metadata, read_metrics, read_run, git_info, slurm_info,
)
from src.training.trainer import get_model, train_model, make_autocast


def _config(tmp_path, **training_overrides):
    cfg = {
        "model": {"architecture": "tinyvgg", "num_classes": 10, "pretrained": False},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 1e-3,
                     "weight_decay": 0.0, "optimizer": "adam", "scheduler": "cosine",
                     "early_stopping_patience": 10, "seed": 0, **training_overrides},
        "augmentation": {"enabled": False},
        "monitoring": {"mlflow_tracking": False, "wandb_enabled": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return Config(str(path))


def _loader(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8)


class TestExperimentLogger:
    def test_writes_run_json_and_metrics(self, tmp_path):
        with ExperimentLogger(str(tmp_path), "run-a", config={"a": 1}) as ex:
            ex.log_params({"lr": 0.1})
            ex.log_metrics({"val_acc": 0.5}, step=1)
            ex.log_metrics({"val_acc": 0.6}, step=2)
            ex.finish({"test_acc": 0.7})

        run = read_run(str(tmp_path))
        assert run["run_name"] == "run-a"
        assert run["params"] == {"lr": 0.1}
        assert run["summary"] == {"test_acc": 0.7}
        assert run["status"] == "finished"
        assert run["config"] == {"a": 1}
        assert "duration_sec" in run and "git" in run and "device" in run

        metrics = read_metrics(str(tmp_path))
        assert [m["step"] for m in metrics] == [1, 2]
        assert metrics[-1]["val_acc"] == 0.6

    def test_context_manager_marks_failure(self, tmp_path):
        with pytest.raises(RuntimeError):
            with ExperimentLogger(str(tmp_path), "run-b"):
                raise RuntimeError("boom")
        assert read_run(str(tmp_path))["status"] == "failed"

    def test_missing_trackers_do_not_break(self, tmp_path, monkeypatch):
        # Even if mlflow/wandb aren't installed (or fail), the logger must work.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name in ("mlflow", "wandb"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        ex = ExperimentLogger(str(tmp_path), "run-c", use_mlflow=True, use_wandb=True)
        ex.log_metrics({"x": 1.0}, step=0)
        ex.finish()
        assert read_metrics(str(tmp_path))[0]["x"] == 1.0

    def test_metadata_shape(self):
        meta = collect_run_metadata({"k": "v"}, torch.device("cpu"), argv=["train.py"])
        assert meta["config"] == {"k": "v"}
        assert meta["argv"] == ["train.py"]
        assert set(meta["git"]) == {"commit", "branch", "dirty"}
        assert isinstance(slurm_info(), dict)

    def test_slurm_env_is_captured(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
        info = slurm_info()
        assert info["slurm_job_id"] == "123" and info["slurm_array_task_id"] == "7"


class TestTrainModelExtras:
    def test_run_artifacts_written(self, tmp_path):
        cfg = _config(tmp_path)
        loader = _loader()
        hist = train_model(get_model("tinyvgg", config=cfg), loader, loader, loader,
                           cfg, torch.device("cpu"), str(tmp_path / "out"), "tinyvgg",
                           run_name="tv-s0")
        run_dir = tmp_path / "out" / "tinyvgg"
        assert (run_dir / "tinyvgg_last.pt").exists()
        run = read_run(str(run_dir))
        assert run["run_name"] == "tv-s0"
        assert run["params"]["model"] == "tinyvgg"
        assert run["params"]["train_samples"] == 24
        assert run["summary"]["test_acc"] == hist["test_acc"]
        assert len(read_metrics(str(run_dir))) == 1
        assert len(hist["epoch_time_sec"]) == 1

    def test_resume_continues_epoch_count(self, tmp_path):
        loader = _loader()
        out = str(tmp_path / "out")
        cfg1 = _config(tmp_path, epochs=1)
        train_model(get_model("tinyvgg", config=cfg1), loader, loader, None,
                    cfg1, torch.device("cpu"), out, "tinyvgg")

        cfg3 = _config(tmp_path, epochs=3)
        hist = train_model(get_model("tinyvgg", config=cfg3), loader, loader, loader,
                           cfg3, torch.device("cpu"), out, "tinyvgg", resume=True)
        assert len(hist["train_loss"]) == 3          # 1 old + 2 new epochs
        state = torch.load(tmp_path / "out" / "tinyvgg" / "tinyvgg_last.pt", weights_only=False)
        assert state["epoch"] == 2
        # Cosine schedule was rebuilt for 3 epochs: the resumed epoch (index 1)
        # must not run at the old T_max=1 horizon's LR of exactly 0.
        assert hist["learning_rates"][1] > 0.0
        assert hist["learning_rates"][1] == pytest.approx(1e-3 * 0.5 * (1 + __import__("math").cos(__import__("math").pi * 1 / 3)), rel=1e-6)

    def test_resume_without_state_starts_fresh(self, tmp_path):
        loader = _loader()
        cfg = _config(tmp_path, epochs=2)
        hist = train_model(get_model("tinyvgg", config=cfg), loader, loader, None,
                           cfg, torch.device("cpu"), str(tmp_path / "o"), "tinyvgg", resume=True)
        assert len(hist["train_loss"]) == 2

    def test_amp_on_cpu_uses_bf16(self, tmp_path):
        cfg = _config(tmp_path, amp=True, grad_clip=1.0, label_smoothing=0.1, optimizer="adamw")
        loader = _loader()
        hist = train_model(get_model("tinyvgg", config=cfg), loader, loader, loader,
                           cfg, torch.device("cpu"), str(tmp_path / "o"), "tinyvgg")
        assert hist["amp"] == "bf16"
        assert 0.0 <= hist["test_acc"] <= 1.0

    def test_make_autocast_disabled(self):
        ctx, scaler, dtype = make_autocast(torch.device("cpu"), enabled=False)
        assert scaler is None and dtype == "fp32"
        with ctx():
            pass

    def test_bad_optimizer_name(self, tmp_path):
        cfg = _config(tmp_path, optimizer="rmsprop")
        loader = _loader()
        with pytest.raises(ValueError):
            train_model(get_model("tinyvgg", config=cfg), loader, loader, None,
                        cfg, torch.device("cpu"), str(tmp_path / "o"), "tinyvgg")
