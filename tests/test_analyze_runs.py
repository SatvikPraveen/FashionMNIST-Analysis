"""
Tests for the batch analysis CLI, on a real TinyVGG checkpoint and synthetic data.
"""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cli import analyze_runs as ar
from src.models.registry import ModelSpec, build_from_spec, spec_path_for


def _make_run(root, name, group, seed, status="finished"):
    run_dir = root / name / "tinyvgg"
    run_dir.mkdir(parents=True)
    spec = ModelSpec(name="tinyvgg")
    torch.manual_seed(seed)
    model = build_from_spec(spec)
    weights = run_dir / "tinyvgg_best.pth"
    torch.save(model.state_dict(), weights)
    spec.save(spec_path_for(str(weights)))
    (run_dir / "run.json").write_text(json.dumps({
        "status": status, "params": {"model": "tinyvgg", "seed": seed},
        "config": {"sweep": {"group": group, "variant": group.split("|")[-1]}},
    }))
    return run_dir


def _loaders(seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(48, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (48,), generator=g)
    dl = DataLoader(TensorDataset(x, y), batch_size=16)
    return dl, dl


def test_find_finished_runs_skips_failed(tmp_path):
    _make_run(tmp_path, "a", "tinyvgg|full", 0)
    _make_run(tmp_path, "b", "tinyvgg|full", 1, status="failed")
    assert [p.parent.name for p in ar.find_finished_runs([str(tmp_path)])] == ["a"]


def test_analyze_and_summarize(tmp_path):
    dirs = [_make_run(tmp_path, f"r{s}", "tinyvgg|full", s) for s in (0, 1)]
    dirs.append(_make_run(tmp_path, "n0", "tinyvgg|no_crop", 0))
    rows = [ar.analyze_run(d, _loaders, torch.device("cpu"), robustness=False) for d in dirs]
    for d, r in zip(dirs, rows):
        assert (d / "analysis" / "analysis.json").exists()
        assert 0.0 <= r["accuracy"] <= 1.0 and r["temperature"] > 0
        assert r["ece_ts"] is not None
    summary = {s["group"]: s for s in ar.summarize(rows)}
    assert summary["tinyvgg|full"]["n"] == 2
    assert summary["tinyvgg|no_crop"]["accuracy_std"] == 0.0
    md = ar.to_markdown(list(summary.values()))
    assert "ECE after T" in md and "corruption" not in md


def test_skip_existing_reuses_report(tmp_path):
    d = _make_run(tmp_path, "r0", "g", 0)
    first = ar.analyze_run(d, _loaders, torch.device("cpu"), robustness=False)

    def boom(seed):
        raise AssertionError("loaders must not be rebuilt when analysis.json exists")
    again = ar.analyze_run(d, boom, torch.device("cpu"), robustness=False, skip_existing=True)
    assert again["accuracy"] == first["accuracy"]


def test_robustness_columns(tmp_path, monkeypatch):
    d = _make_run(tmp_path, "r0", "g", 0)
    import src.evaluation.analysis as an
    orig = an.robustness_sweep

    def small_sweep(model, loader, device, *args, **kwargs):
        return orig(model, loader, device, corruptions=["contrast"], severities=(1,),
                    clean_accuracy=kwargs.get("clean_accuracy"))
    monkeypatch.setattr(an, "robustness_sweep", small_sweep)
    row = ar.analyze_run(d, _loaders, torch.device("cpu"), robustness=True)
    assert "mean_corruption_error" in row and "err_contrast" in row
    assert "corruption" in ar.to_markdown(ar.summarize([row]))
