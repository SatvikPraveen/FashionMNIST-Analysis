"""
Research-grade evaluation beyond accuracy.

Pure functions over (logits, labels) so every metric is unit-testable
without a dataset, plus a ``run_full_analysis`` driver that writes JSON and
figures for a trained model.

Covers:
    * Calibration: expected / maximum calibration error, NLL, Brier score,
      reliability diagram, post-hoc temperature scaling (fit on validation).
    * Per-class precision / recall / F1 and the most confused class pairs
      (Fashion-MNIST's known hard cluster: T-shirt / Shirt / Pullover / Coat).
    * Robustness: accuracy under synthetic corruptions (Gaussian noise, blur,
      contrast, brightness, rotation, occlusion, translation) at 5
      severities, with mean corruption error.

Usage:
    from src.evaluation.analysis import run_full_analysis
    report = run_full_analysis(model, test_loader, device, out_dir="results/analysis")
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def collect_outputs(model: nn.Module, loader, device: torch.device,
                    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader; returns (logits, labels) on CPU."""
    model.eval()
    logits, labels = [], []
    for X, y in loader:
        X = X.to(device)
        if transform is not None:
            X = transform(X)
        logits.append(model(X).float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def calibration_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> List[Dict[str, float]]:
    """Per-bin confidence / accuracy / count for a reliability diagram."""
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        bins.append({
            "lower": lo, "upper": hi, "count": n,
            "confidence": float(conf[mask].mean()) if n else float("nan"),
            "accuracy": float(correct[mask].mean()) if n else float("nan"),
        })
    return bins


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
                               ) -> Dict[str, float]:
    """ECE (count-weighted |acc - conf|) and MCE (max over bins)."""
    bins = calibration_bins(probs, labels, n_bins)
    total = sum(b["count"] for b in bins) or 1
    ece = sum(b["count"] / total * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"])
    mce = max((abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]), default=0.0)
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins}


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels))


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    onehot = F.one_hot(labels, probs.shape[1]).float()
    return float(((probs - onehot) ** 2).sum(dim=1).mean())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al. 2017): minimise NLL of logits / T."""
    # Outputs collected under inference_mode are "inference tensors" and
    # cannot take part in autograd; cloning outside that mode fixes it.
    logits = logits.detach().clone().float()
    labels = labels.detach().clone()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calibration_report(logits: torch.Tensor, labels: torch.Tensor,
                       temperature: Optional[float] = None, n_bins: int = 15) -> Dict[str, object]:
    """All calibration metrics before (and optionally after) temperature scaling."""
    probs = logits.softmax(dim=1)
    out: Dict[str, object] = {
        "accuracy": float((probs.argmax(1) == labels).float().mean()),
        "nll": negative_log_likelihood(logits, labels),
        "brier": brier_score(probs, labels),
        **expected_calibration_error(probs, labels, n_bins),
        "bins": calibration_bins(probs, labels, n_bins),
        "mean_confidence": float(probs.max(1).values.mean()),
    }
    if temperature is not None:
        scaled = (logits / temperature).softmax(dim=1)
        out["temperature"] = temperature
        out["scaled"] = {
            "nll": negative_log_likelihood(logits / temperature, labels),
            "brier": brier_score(scaled, labels),
            **expected_calibration_error(scaled, labels, n_bins),
            "bins": calibration_bins(scaled, labels, n_bins),
            "mean_confidence": float(scaled.max(1).values.mean()),
        }
    return out


# --------------------------------------------------------------------------- #
# Per-class analysis
# --------------------------------------------------------------------------- #

def per_class_report(preds: torch.Tensor, labels: torch.Tensor,
                     class_names: Sequence[str] = CLASS_NAMES, top_confusions: int = 8
                     ) -> Dict[str, object]:
    k = len(class_names)
    cm = torch.zeros(k, k, dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t, p] += 1
    classes = []
    for c in range(k):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum()) - tp; fn = int(cm[c].sum()) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        classes.append({"class": class_names[c], "index": c, "support": int(cm[c].sum()),
                        "precision": prec, "recall": rec, "f1": f1})
    off = [(int(cm[t, p]), t, p) for t in range(k) for p in range(k) if t != p and cm[t, p] > 0]
    off.sort(reverse=True)
    confusions = [{"true": class_names[t], "pred": class_names[p], "count": n,
                   "rate": n / int(cm[t].sum())} for n, t, p in off[:top_confusions]]
    return {"classes": classes, "macro_f1": sum(c["f1"] for c in classes) / k,
            "confusion_matrix": cm.tolist(), "top_confusions": confusions}


# --------------------------------------------------------------------------- #
# Robustness to corruptions
# --------------------------------------------------------------------------- #

def _denorm(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return x * std + mean


def _renorm(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x.clamp(0, 1) - mean) / std


def _gaussian_kernel(sigma: float, size: int = 5) -> torch.Tensor:
    ax = torch.arange(size) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def c_gaussian_noise(x, s):  # x in [0,1]
    std = [0.04, 0.08, 0.12, 0.18, 0.26][s - 1]
    return x + torch.randn_like(x) * std


def c_gaussian_blur(x, s):
    sigma = [0.4, 0.6, 0.8, 1.0, 1.3][s - 1]
    k = _gaussian_kernel(sigma).to(x)
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k)


def c_contrast(x, s):
    f = [0.75, 0.6, 0.45, 0.3, 0.2][s - 1]
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * f + m


def c_brightness(x, s):
    d = [0.1, 0.2, 0.3, 0.4, 0.5][s - 1]
    return x + d


def c_rotation(x, s):
    deg = [5, 10, 15, 25, 35][s - 1]
    th = math.radians(deg)
    mat = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                        [math.sin(th), math.cos(th), 0.0]], dtype=x.dtype, device=x.device)
    grid = F.affine_grid(mat.expand(x.shape[0], 2, 3), x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def c_translation(x, s):
    px = [1, 2, 3, 4, 6][s - 1]
    return torch.roll(x, shifts=(px, px), dims=(2, 3))


def c_occlusion(x, s):
    size = [4, 6, 8, 10, 12][s - 1]
    out = x.clone()
    h, w = x.shape[-2:]
    g = torch.Generator(device="cpu").manual_seed(1234 + s)
    ys = torch.randint(0, h - size + 1, (x.shape[0],), generator=g)
    xs = torch.randint(0, w - size + 1, (x.shape[0],), generator=g)
    for i in range(x.shape[0]):
        out[i, :, ys[i]:ys[i] + size, xs[i]:xs[i] + size] = 0.0
    return out


CORRUPTIONS: Dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "gaussian_noise": c_gaussian_noise,
    "gaussian_blur": c_gaussian_blur,
    "contrast": c_contrast,
    "brightness": c_brightness,
    "rotation": c_rotation,
    "translation": c_translation,
    "occlusion": c_occlusion,
}


def make_corruption(name: str, severity: int, mean: float = FMNIST_MEAN, std: float = FMNIST_STD
                    ) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a transform on *normalised* batches applying the corruption in pixel space."""
    fn = CORRUPTIONS[name]

    def apply(x: torch.Tensor) -> torch.Tensor:
        return _renorm(fn(_denorm(x, mean, std), severity), mean, std)
    return apply


def robustness_sweep(model: nn.Module, loader, device: torch.device,
                     corruptions: Iterable[str] = tuple(CORRUPTIONS),
                     severities: Sequence[int] = (1, 2, 3, 4, 5),
                     clean_accuracy: Optional[float] = None,
                     seed: int = 0) -> Dict[str, object]:
    """Accuracy per (corruption, severity) plus mean corruption error (mCE-style, unnormalised)."""
    if clean_accuracy is None:
        logits, labels = collect_outputs(model, loader, device)
        clean_accuracy = float((logits.argmax(1) == labels).float().mean())
    results: Dict[str, Dict[str, float]] = {}
    for name in corruptions:
        results[name] = {}
        for s in severities:
            torch.manual_seed(seed + s)  # noise corruptions are stochastic
            logits, labels = collect_outputs(model, loader, device, make_corruption(name, s))
            results[name][str(s)] = float((logits.argmax(1) == labels).float().mean())
    per_corruption_err = {n: 1.0 - sum(v.values()) / len(v) for n, v in results.items()}
    mce = sum(per_corruption_err.values()) / len(per_corruption_err) if per_corruption_err else float("nan")
    return {"clean_accuracy": clean_accuracy, "accuracy": results,
            "mean_error_per_corruption": per_corruption_err, "mean_corruption_error": mce,
            "relative_mce": mce - (1.0 - clean_accuracy)}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_reliability(bins: List[Dict[str, float]], path: str, title: str = "Reliability diagram") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [(b["lower"] + b["upper"]) / 2 for b in bins]
    width = bins[0]["upper"] - bins[0]["lower"]
    accs = [b["accuracy"] if b["count"] else 0.0 for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(centers, accs, width=width, edgecolor="black", alpha=0.8, label="accuracy")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_robustness(rob: Dict[str, object], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, sev in rob["accuracy"].items():
        xs = [int(k) for k in sev]; ys = [sev[k] for k in sev]
        ax.plot(xs, ys, marker="o", label=name)
    ax.axhline(rob["clean_accuracy"], color="gray", ls="--", label="clean")
    ax.set_xlabel("severity"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
    ax.set_title("Accuracy under corruption"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_per_class(report: Dict[str, object], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [c["class"] for c in report["classes"]]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, key in enumerate(("precision", "recall", "f1")):
        ax.bar(x + (i - 1) * w, [c[key] for c in report["classes"]], w, label=key)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_title(f"Per-class metrics (macro-F1 {report['macro_f1']:.4f})"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_full_analysis(model: nn.Module, test_loader, device: torch.device, out_dir: str,
                      val_loader=None, class_names: Sequence[str] = CLASS_NAMES,
                      robustness: bool = True, corruptions: Optional[Iterable[str]] = None,
                      severities: Sequence[int] = (1, 2, 3, 4, 5), n_bins: int = 15,
                      figures_dir: Optional[str] = None) -> Dict[str, object]:
    """Compute everything, write ``analysis.json`` + figures, return the report."""
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = figures_dir or out_dir
    os.makedirs(figures_dir, exist_ok=True)

    logits, labels = collect_outputs(model, test_loader, device)
    preds = logits.argmax(1)

    temperature = None
    if val_loader is not None:
        v_logits, v_labels = collect_outputs(model, val_loader, device)
        temperature = fit_temperature(v_logits, v_labels)

    report: Dict[str, object] = {
        "n_test": int(labels.numel()),
        "accuracy": float((preds == labels).float().mean()),
        "calibration": calibration_report(logits, labels, temperature, n_bins),
        "per_class": per_class_report(preds, labels, class_names),
    }
    plot_reliability(report["calibration"]["bins"], os.path.join(figures_dir, "reliability_diagram.png"))
    if temperature is not None:
        plot_reliability(report["calibration"]["scaled"]["bins"],
                         os.path.join(figures_dir, "reliability_diagram_temperature_scaled.png"),
                         title=f"Reliability (T={temperature:.3f})")
    plot_per_class(report["per_class"], os.path.join(figures_dir, "per_class_metrics.png"))

    if robustness:
        report["robustness"] = robustness_sweep(model, test_loader, device,
                                                corruptions or tuple(CORRUPTIONS), severities,
                                                clean_accuracy=report["accuracy"])
        plot_robustness(report["robustness"], os.path.join(figures_dir, "robustness.png"))

    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def summarize_report(report: Dict[str, object]) -> str:
    """Short human-readable summary of a run_full_analysis report."""
    cal = report["calibration"]
    lines = [f"accuracy      {report['accuracy']:.4f}  (n={report['n_test']})",
             f"macro-F1      {report['per_class']['macro_f1']:.4f}",
             f"NLL / Brier   {cal['nll']:.4f} / {cal['brier']:.4f}",
             f"ECE / MCE     {cal['ece']:.4f} / {cal['mce']:.4f}"]
    if "scaled" in cal:
        s = cal["scaled"]
        lines.append(f"  after T={cal['temperature']:.3f}: NLL {s['nll']:.4f}  ECE {s['ece']:.4f}")
    if "robustness" in report:
        r = report["robustness"]
        lines.append(f"mean corruption error {r['mean_corruption_error']:.4f} "
                     f"(relative {r['relative_mce']:+.4f})")
        worst = max(r["mean_error_per_corruption"].items(), key=lambda kv: kv[1])
        lines.append(f"  worst corruption: {worst[0]} ({worst[1]:.4f} error)")
    top = report["per_class"]["top_confusions"][:3]
    if top:
        lines.append("top confusions: " + "; ".join(f"{c['true']}->{c['pred']} {c['count']}" for c in top))
    return "\n".join(lines)
