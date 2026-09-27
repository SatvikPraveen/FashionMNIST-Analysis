"""
Tests for seed ensembling (synthetic probabilities + one real checkpoint round-trip).
"""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cli import ensemble_runs as er
from src.models.registry import ModelSpec, build_from_spec, spec_path_for


def test_ensemble_fixes_independent_errors():
    # 3 members, each wrong on a *different* third of the data -> majority is always right
    n = 300
    labels = torch.randint(0, 10, (n,))
    probs = []
    for m in range(3):
        p = torch.nn.functional.one_hot(labels, 10).float() * 0.9 + 0.01
        wrong = torch.zeros(n, dtype=torch.bool)
        wrong[m * 100:(m + 1) * 100] = True
        bad = torch.nn.functional.one_hot((labels + 1) % 10, 10).float() * 0.6 + 0.04
        p[wrong] = bad[wrong]
        probs.append(p / p.sum(1, keepdim=True))
    out = er.ensemble_group(probs, labels)
    assert out["member_accuracy_mean"] == pytest.approx(2 / 3)
    assert out["ensemble_accuracy"] == pytest.approx(1.0)
    assert out["accuracy_gain"] == pytest.approx(1 / 3)
    assert out["disagreement"] == pytest.approx(1.0)
    assert out["ensemble_nll"] < out["member_nll_mean"]


def test_identical_members_change_nothing():
    labels = torch.randint(0, 10, (50,))
    p = torch.rand(50, 10).softmax(1)
    out = er.ensemble_group([p, p.clone()], labels)
    assert out["accuracy_gain"] == pytest.approx(0.0)
    assert out["disagreement"] == 0.0
    assert out["ensemble_ece"] == pytest.approx(out["member_ece_mean"])


def test_group_runs_and_member_probs(tmp_path):
    dirs = []
    for s in (0, 1):
        d = tmp_path / f"r{s}" / "tinyvgg"; d.mkdir(parents=True)
        spec = ModelSpec(name="tinyvgg"); torch.manual_seed(s)
        w = d / "tinyvgg_best.pth"
        torch.save(build_from_spec(spec).state_dict(), w); spec.save(spec_path_for(str(w)))
        (d / "run.json").write_text(json.dumps({"status": "finished", "params": {"model": "tinyvgg", "seed": s},
                                                 "config": {"sweep": {"group": "tinyvgg|full"}}}))
        dirs.append(d)
    groups = er.group_runs(dirs)
    assert list(groups) == ["tinyvgg|full"] and len(groups["tinyvgg|full"]) == 2
    loader = DataLoader(TensorDataset(torch.randn(20, 1, 28, 28), torch.randint(0, 10, (20,))), batch_size=8)
    probs = [er.member_probs(d, loader, torch.device("cpu")) for d in dirs]
    assert probs[0].shape == (20, 10)
    assert torch.allclose(probs[0].sum(1), torch.ones(20), atol=1e-5)
    md = er.to_markdown([{"group": "g", **er.ensemble_group(probs, torch.randint(0, 10, (20,)))}])
    assert "ensemble acc" in md
