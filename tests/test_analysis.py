"""
Tests for calibration, per-class and robustness analysis (synthetic data only).
"""

import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import analysis as an
from src.models.architectures import MiniCNN


def _perfectly_calibrated(n=4000, seed=0):
    """Confidence c on the predicted class; correct with probability c."""
    g = torch.Generator().manual_seed(seed)
    conf = torch.rand(n, generator=g) * 0.5 + 0.5           # in [0.5, 1]
    labels = torch.randint(0, 10, (n,), generator=g)
    correct = torch.rand(n, generator=g) < conf
    preds = torch.where(correct, labels, (labels + 1) % 10)
    probs = torch.full((n, 10), 0.0)
    probs[torch.arange(n), preds] = conf
    other = (1 - conf) / 9
    probs = torch.where(probs == 0, other[:, None].expand(n, 10), probs)
    return probs, labels


class TestCalibration:
    def test_ece_near_zero_when_calibrated(self):
        probs, labels = _perfectly_calibrated()
        m = an.expected_calibration_error(probs, labels, n_bins=10)
        assert m["ece"] < 0.03

    def test_ece_large_when_overconfident(self):
        n = 1000
        labels = torch.randint(0, 10, (n,))
        probs = torch.full((n, 10), 0.01)
        probs[torch.arange(n), (labels + 1) % 10] = 0.91   # confident and always wrong
        m = an.expected_calibration_error(probs, labels)
        assert m["ece"] == pytest.approx(0.91, abs=1e-6)
        assert m["mce"] == pytest.approx(0.91, abs=1e-6)

    def test_bins_cover_all_samples(self):
        probs, labels = _perfectly_calibrated(n=500)
        bins = an.calibration_bins(probs, labels, n_bins=7)
        assert sum(b["count"] for b in bins) == 500

    def test_brier_and_nll_bounds(self):
        logits = torch.randn(64, 10)
        labels = torch.randint(0, 10, (64,))
        probs = logits.softmax(1)
        assert 0.0 <= an.brier_score(probs, labels) <= 2.0
        assert an.negative_log_likelihood(logits, labels) > 0

    def test_temperature_scaling_reduces_nll(self):
        torch.manual_seed(0)
        n = 2000
        labels = torch.randint(0, 10, (n,))
        # Model is right 70% of the time but always very confident -> overconfident
        wrong = torch.rand(n) < 0.3
        pred = torch.where(wrong, (labels + 1) % 10, labels)
        overconfident = torch.nn.functional.one_hot(pred, 10).float() * 12.0
        T = an.fit_temperature(overconfident, labels)
        assert T > 2.0
        before = an.negative_log_likelihood(overconfident, labels)
        after = an.negative_log_likelihood(overconfident / T, labels)
        assert after < before
        rep = an.calibration_report(overconfident, labels, temperature=T)
        assert rep["scaled"]["ece"] < rep["ece"]


class TestPerClass:
    def test_metrics_and_confusions(self):
        labels = torch.tensor([0, 0, 0, 6, 6, 6, 1, 1])
        preds = torch.tensor([0, 0, 6, 6, 0, 0, 1, 1])
        rep = an.per_class_report(preds, labels)
        by = {c["class"]: c for c in rep["classes"]}
        assert by["Trouser"]["f1"] == 1.0
        assert by["T-shirt/top"]["recall"] == pytest.approx(2 / 3)
        assert by["T-shirt/top"]["precision"] == pytest.approx(2 / 4)
        assert rep["top_confusions"][0] == {"true": "Shirt", "pred": "T-shirt/top", "count": 2, "rate": 2 / 3}
        assert sum(sum(r) for r in rep["confusion_matrix"]) == 8


class TestRobustness:
    def test_corruptions_preserve_shape_and_change_input(self):
        x = torch.rand(4, 1, 28, 28)
        xn = (x - an.FMNIST_MEAN) / an.FMNIST_STD
        for name in an.CORRUPTIONS:
            for s in (1, 5):
                y = an.make_corruption(name, s)(xn)
                assert y.shape == xn.shape, name
                assert torch.isfinite(y).all(), name
                assert not torch.allclose(y, xn), f"{name} s={s} was a no-op"

    def test_severity_is_monotone_for_noise(self):
        x = torch.rand(8, 1, 28, 28)
        d = [(an.c_gaussian_noise(x, s) - x).abs().mean().item() for s in range(1, 6)]
        assert d == sorted(d)

    def test_sweep_and_full_analysis(self, tmp_path):
        torch.manual_seed(0)
        model = MiniCNN(in_channels=1, num_classes=10).eval()
        x = torch.randn(40, 1, 28, 28); y = torch.randint(0, 10, (40,))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        rob = an.robustness_sweep(model, loader, torch.device("cpu"),
                                  corruptions=["contrast", "occlusion"], severities=(1, 3))
        assert set(rob["accuracy"]) == {"contrast", "occlusion"}
        assert set(rob["accuracy"]["contrast"]) == {"1", "3"}
        assert 0 <= rob["mean_corruption_error"] <= 1

        report = an.run_full_analysis(model, loader, torch.device("cpu"), str(tmp_path),
                                      val_loader=loader, corruptions=["brightness"], severities=(2,))
        saved = json.load(open(tmp_path / "analysis.json"))
        assert saved["n_test"] == 40 and "temperature" in saved["calibration"]
        for f in ("reliability_diagram.png", "reliability_diagram_temperature_scaled.png",
                  "per_class_metrics.png", "robustness.png"):
            assert (tmp_path / f).exists(), f
        text = an.summarize_report(report)
        assert "ECE" in text and "mean corruption error" in text
