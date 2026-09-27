#!/usr/bin/env python3
"""
Build the project website from the committed sweep results.

Reads only ``results/sweeps/*.csv`` (standard library, no third-party
packages) and writes a single self-contained ``index.html`` with inline SVG
charts, a table view for every chart, hover/focus tooltips and light/dark
themes. The GitHub Pages workflow runs this on every push to main, so the
site can never drift from the data in the repository.

Usage:
    python site/build.py                 # -> _site/index.html
    python site/build.py --out /tmp/site
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "sweeps"
REPO_URL = "https://github.com/SatvikPraveen/FashionMNIST-Analysis"

# Categorical slots 1-3 of the validated reference palette (light / dark).
SERIES = {
    "s1": ("#2a78d6", "#3987e5"),
    "s2": ("#eb6834", "#d95926"),
    "s3": ("#1baf7a", "#199e70"),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def read_csv(name: str) -> List[Dict[str, str]]:
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def row(rows: List[Dict[str, str]], **match) -> Dict[str, str]:
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            return r
    raise KeyError(f"no row matching {match}")


def f(x) -> float:
    return float(x)


def count_runs() -> int:
    total = 0
    for p in RESULTS.glob("*_runs.csv"):
        if "analysis" in p.name or "_vs_" in p.name:
            continue
        with open(p, newline="") as fh:
            total += sum(1 for _ in csv.DictReader(fh))
    return total


def load() -> Dict[str, object]:
    base = read_csv("baseline_fixed_summary.csv")
    aug = read_csv("augmentation_fixed_summary.csv")
    cropgen = read_csv("crop_generalization_summary.csv")
    bb_fixed = read_csv("backbones_fixed_summary.csv")
    bb = read_csv("backbones_summary.csv")
    ens = read_csv("augmentation_fixed_ensemble.csv")

    def model_pt(label, r, cat, note=""):
        return {"label": label, "cat": cat, "mean": f(r["test_acc_mean"]),
                "ci": f(r["test_acc_ci95"]), "std": f(r["test_acc_std"]), "n": int(r["n"]),
                "params": int(float(r["num_parameters"])), "note": note}

    tv_nc_ens = row(ens, group="tinyvgg|no_crop")
    models = [
        model_pt("ViT-Tiny/16", row(bb_fixed, group="vit_tiny|pretrained"), "pretrained"),
        model_pt("ConvNeXt-Tiny", row(bb_fixed, group="convnext_tiny|pretrained"), "pretrained"),
        model_pt("EfficientNet-B0", row(bb, group="efficientnet_b0|pretrained"), "pretrained", "pre-fix pipeline"),
        model_pt("ResNet-18 (timm)", row(bb, group="resnet18|pretrained"), "pretrained", "pre-fix pipeline"),
        {"label": "TinyVGG, no crop, 5-seed ensemble", "cat": "ensemble",
         "mean": f(tv_nc_ens["ensemble_accuracy"]), "ci": None, "std": None, "n": 5,
         "params": 5 * 142794, "note": "one ensemble of 5 members"},
        model_pt("ResNet-18 (custom)", row(base, group="resnet|default"), "scratch"),
        model_pt("TinyVGG, no crop", row(aug, group="tinyvgg|no_crop"), "scratch"),
        model_pt("TinyVGG", row(base, group="tinyvgg|default"), "scratch"),
        model_pt("MiniCNN, no crop", row(cropgen, group="minicnn|no_crop"), "scratch"),
        model_pt("MiniCNN", row(base, group="minicnn|default"), "scratch"),
    ]
    models.sort(key=lambda m: -m["mean"])

    def crop_pt(label, fname, group, params):
        r = row(read_csv(fname), group=group)
        n = int(r["n_paired"])
        return {"label": label, "params": params, "diff": f(r["diff_mean"]),
                "ci": f(r["diff_ci95"]), "p": f(r["p_value"]), "wins": int(r["wins"]), "n": n}

    crop = [
        crop_pt("MiniCNN", "crop_generalization_vs_minicnn_paired.csv", "minicnn|no_crop", 105866),
        crop_pt("TinyVGG", "augmentation_fixed_paired.csv", "tinyvgg|no_crop", 142794),
        crop_pt("ViT-Tiny/16 (pretrained)", "backbones_fixed_vs_vit_tiny_paired.csv",
                "vit_tiny|pretrained_no_crop", 5428042),
        crop_pt("ResNet-18 (custom)", "crop_generalization_vs_resnet_paired.csv", "resnet|no_crop", 11172810),
        crop_pt("ConvNeXt-Tiny (pretrained)", "backbones_fixed_vs_convnext_tiny_paired.csv",
                "convnext_tiny|pretrained_no_crop", 27824746),
    ]

    pretrain = []
    for key, label in [("vit_tiny", "ViT-Tiny/16"), ("convnext_tiny", "ConvNeXt-Tiny"),
                       ("efficientnet_b0", "EfficientNet-B0"), ("resnet18", "ResNet-18 (timm)")]:
        p = row(bb, group=f"{key}|pretrained"); s = row(bb, group=f"{key}|scratch")
        pretrain.append({"label": label, "pre": f(p["test_acc_mean"]), "pre_ci": f(p["test_acc_ci95"]),
                         "scr": f(s["test_acc_mean"]), "scr_ci": f(s["test_acc_ci95"]),
                         "params": int(float(p["num_parameters"]))})
    pretrain.sort(key=lambda d: -(d["pre"] - d["scr"]))

    variant_names = {"no_crop": "no random crop", "no_rotation": "no rotation", "full": "full recipe",
                     "legacy": "pre-fix pipeline", "no_aug": "no augmentation",
                     "no_mixup_cutmix": "no Mixup/CutMix", "no_flip": "no horizontal flip"}
    calib = []
    for r in ens:
        v = r["group"].split("|")[-1]
        mixup = v not in ("no_aug", "no_mixup_cutmix")
        calib.append({"label": f"TinyVGG, {variant_names.get(v, v)}", "mixup": mixup,
                      "member": f(r["member_ece_mean"]), "ensemble": f(r["ensemble_ece"]),
                      "acc_member": f(r["member_accuracy_mean"]), "acc_ens": f(r["ensemble_accuracy"])})
    calib.sort(key=lambda d: (not d["mixup"], -d["ensemble"]))

    best = max(models, key=lambda m: m["mean"])
    best_scratch = max((m for m in models if m["cat"] == "scratch"), key=lambda m: m["mean"])
    return {"models": models, "crop": crop, "pretrain": pretrain, "calib": calib,
            "best": best, "best_scratch": best_scratch, "n_runs": count_runs()}


# --------------------------------------------------------------------------- #
# SVG helpers
# --------------------------------------------------------------------------- #

E = html.escape


class Geo:
    """Chart geometry. 'wide': labels in a left column. 'narrow': labels above each row."""

    def __init__(self, narrow: bool):
        self.narrow = narrow
        self.W = 400 if narrow else 760
        self.LEFT = 14 if narrow else 230
        self.RIGHT = 44 if narrow else 36
        self.ROW = 50 if narrow else 34
        self.TOP = 22 if narrow else 16
        self.AXIS = 46

    def row_y(self, i: int) -> float:
        """Vertical position of the marks in row i."""
        base = self.TOP + self.ROW * i
        return base + (self.ROW - 14 if self.narrow else self.ROW / 2)

    def label(self, i: int, text: str, sub: str = "") -> str:
        if self.narrow:
            y = self.TOP + self.ROW * i + 16
            extra = f'<tspan class="sublabel" dx="6">{E(sub)}</tspan>' if sub else ""
            return f'<text class="label" x="{self.LEFT}" y="{y:.1f}">{E(text)}{extra}</text>'
        y = self.row_y(i)
        if sub:
            return (f'<text class="label" x="{self.LEFT - 12}" y="{y - 2:.1f}" text-anchor="end">{E(text)}</text>'
                    f'<text class="sublabel" x="{self.LEFT - 12}" y="{y + 11:.1f}" text-anchor="end">{E(sub)}</text>')
        return f'<text class="label" x="{self.LEFT - 12}" y="{y + 4:.1f}" text-anchor="end">{E(text)}</text>'

    def height(self, n: int) -> int:
        return int(self.TOP + self.ROW * n + self.AXIS)


def scale(lo: float, hi: float, a: float, b: float):
    return lambda v: a + (v - lo) / (hi - lo) * (b - a)


def ticks(lo: float, hi: float, step: float) -> List[float]:
    out, v = [], lo
    while v <= hi + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


MINUS = "−"


def pct(v: float, digits: int = 2) -> str:
    return f"{v * 100:.{digits}f}%".replace("-", MINUS)


def pts(v: float, digits: int = 2) -> str:
    return f"{v * 100:+.{digits}f}".replace("-", MINUS)


def axis(g: Geo, x, lo, hi, step, y0, y1, fmt, title, zero: Optional[float] = None) -> str:
    parts = []
    for t in ticks(lo, hi, step):
        xx = x(t)
        cls = "zero" if zero is not None and abs(t - zero) < 1e-9 else "grid"
        parts.append(f'<line class="{cls}" x1="{xx:.1f}" x2="{xx:.1f}" y1="{y0}" y2="{y1}"/>')
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{y1 + 18}" text-anchor="middle">{E(fmt(t))}</text>')
    parts.append(f'<text class="axis-title" x="{(x(lo) + x(hi)) / 2:.1f}" y="{y1 + 38}" '
                 f'text-anchor="middle">{E(title)}</text>')
    return "".join(parts)


def svg_open(g: Geo, h: int, label: str) -> str:
    cls = "narrow" if g.narrow else "wide"
    return (f'<svg class="chart {cls}" viewBox="0 0 {g.W} {h}" role="img" aria-label="{E(label)}" '
            f'preserveAspectRatio="xMinYMin meet">')


def hit(g: Geo, i: int, tip: List[str]) -> str:
    """Transparent full-row hit target carrying the tooltip lines."""
    data = E("\n".join(tip))
    y = g.TOP + g.ROW * i
    return (f'<rect class="hit" x="0" y="{y:.1f}" width="{g.W}" height="{g.ROW}" '
            f'tabindex="0" data-tip="{data}"><title>{data}</title></rect>')


def dot(x: float, y: float, cls: str, hollow: bool = False) -> str:
    fill = "var(--surface)" if hollow else f"var(--{cls})"
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="var(--{cls})" '
            f'stroke-width="{2.5 if hollow else 0}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="none" stroke="var(--surface)" stroke-width="2"/>')


def value_label(x_lo: float, x_hi: float, y: float, text: str, negative: bool) -> str:
    """Value beside the whisker end away from zero, so it never sits on the zero line."""
    if negative:
        return f'<text class="value" x="{x_lo - 8:.1f}" y="{y + 4:.1f}" text-anchor="end">{E(text)}</text>'
    return f'<text class="value" x="{x_hi + 8:.1f}" y="{y + 4:.1f}">{E(text)}</text>'


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

CAT = {"scratch": ("s1", "Trained from scratch"), "pretrained": ("s2", "ImageNet-pretrained"),
       "ensemble": ("s3", "Seed ensemble")}


def both(fn, data) -> str:
    """Render a chart in its wide and narrow layouts; CSS shows the one that fits."""
    return fn(Geo(False), data) + fn(Geo(True), data)


def chart_models(g: Geo, models) -> str:
    lo, hi = 0.895, 0.960
    x = scale(lo, hi, g.LEFT, g.W - g.RIGHT)
    y0, y1 = g.TOP, g.TOP + g.ROW * len(models)
    step = 0.02 if g.narrow else 0.01
    out = [svg_open(g, g.height(len(models)), "Test accuracy with 95% confidence intervals by model")]
    out.append(axis(g, x, 0.90, 0.96, step, y0, y1, lambda t: f"{t * 100:.0f}%", "Test accuracy (95% CI)"))
    for i, m in enumerate(models):
        y = g.row_y(i)
        cls, catname = CAT[m["cat"]]
        out.append(g.label(i, m["label"] + (" †" if m["note"] == "pre-fix pipeline" else "")))
        if m["ci"]:
            out.append(f'<line class="whisker" style="stroke:var(--{cls})" x1="{x(m["mean"] - m["ci"]):.1f}" '
                       f'x2="{x(m["mean"] + m["ci"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(dot(x(m["mean"]), y, cls))
        if i == 0:
            hi_x = x(m["mean"] + (m["ci"] or 0))
            if g.narrow:
                out.append(f'<text class="value" x="{x(m["mean"] - (m["ci"] or 0)) - 8:.1f}" y="{y + 4:.1f}" '
                           f'text-anchor="end">{pct(m["mean"])}</text>')
            else:
                out.append(f'<text class="value" x="{hi_x + 10:.1f}" y="{y + 4:.1f}">{pct(m["mean"])}</text>')
        tip = [pct(m["mean"]), m["label"], catname,
               (f"95% CI ±{m['ci'] * 100:.2f} points, n = {m['n']} seeds" if m["ci"] else
                "single ensemble of 5 seed models"),
               f"{m['params'] / 1e6:.2f}M parameters" + (" in total" if m["cat"] == "ensemble" else "")]
        if m["note"] == "pre-fix pipeline":
            tip.append("measured on the pre-fix augmentation pipeline")
        out.append(hit(g, i, tip))
    out.append("</svg>")
    return "".join(out)


def chart_crop(g: Geo, crop) -> str:
    lo, hi = -0.016, 0.025
    x = scale(lo, hi, g.LEFT + (34 if g.narrow else 0), g.W - g.RIGHT)
    y0, y1 = g.TOP, g.TOP + g.ROW * len(crop)
    out = [svg_open(g, g.height(len(crop)), "Accuracy change from removing random crop, by model size")]
    out.append(axis(g, x, -0.01, 0.02, 0.01, y0, y1, lambda t: pts(t, 0) if t else "0",
                    "Change from removing crop, points (95% CI)", zero=0.0))
    out.append(f'<text class="hint" x="{x(0) + 8:.1f}" y="{g.TOP - 6}">crop hurts →</text>')
    out.append(f'<text class="hint" x="{x(0) - 8:.1f}" y="{g.TOP - 6}" text-anchor="end">← crop helps</text>')
    for i, c in enumerate(crop):
        y = g.row_y(i)
        out.append(g.label(i, c["label"], f"{c['params'] / 1e6:.2f}M params"))
        x_lo, x_hi = x(c["diff"] - c["ci"]), x(c["diff"] + c["ci"])
        out.append(f'<line class="whisker" style="stroke:var(--s1)" x1="{x_lo:.1f}" x2="{x_hi:.1f}" '
                   f'y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(dot(x(c["diff"]), y, "s1"))
        out.append(value_label(x_lo, x_hi, y, pts(c["diff"]), c["diff"] < 0))
        out.append(hit(g, i, [f"{pts(c['diff'])} points", c["label"],
                              f"95% CI [{pts(c['diff'] - c['ci'])}, {pts(c['diff'] + c['ci'])}]",
                              f"paired t-test p = {c['p']:.3f}",
                              f"removing crop better on {c['wins']} of {c['n']} seeds"]))
    out.append("</svg>")
    return "".join(out)


def chart_pretrain(g: Geo, pre) -> str:
    lo, hi = 0.78, 0.965
    x = scale(lo, hi, g.LEFT, g.W - g.RIGHT)
    y0, y1 = g.TOP, g.TOP + g.ROW * len(pre)
    out = [svg_open(g, g.height(len(pre)), "Test accuracy from scratch versus ImageNet-pretrained, per backbone")]
    out.append(axis(g, x, 0.80, 0.96, 0.04, y0, y1, lambda t: f"{t * 100:.0f}%", "Test accuracy (mean of 3 seeds)"))
    for i, d in enumerate(pre):
        y = g.row_y(i)
        out.append(g.label(i, d["label"]))
        out.append(f'<line class="connector" x1="{x(d["scr"]):.1f}" x2="{x(d["pre"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(dot(x(d["scr"]), y, "s1"))
        out.append(dot(x(d["pre"]), y, "s2"))
        out.append(f'<text class="value" x="{x(d["pre"]) + 12:.1f}" y="{y + 4:.1f}">{pts(d["pre"] - d["scr"], 1)}</text>')
        out.append(hit(g, i, [f"{pts(d['pre'] - d['scr'], 1)} points from pretraining", d["label"],
                              f"pretrained {pct(d['pre'])} (±{d['pre_ci'] * 100:.2f})",
                              f"from scratch {pct(d['scr'])} (±{d['scr_ci'] * 100:.2f})"]))
    out.append("</svg>")
    return "".join(out)


def chart_calib(g: Geo, cal) -> str:
    lo, hi = 0.0, 0.035
    x = scale(lo, hi, g.LEFT + (6 if g.narrow else 0), g.W - g.RIGHT)
    y0, y1 = g.TOP, g.TOP + g.ROW * len(cal)
    step = 0.01 if g.narrow else 0.005
    out = [svg_open(g, g.height(len(cal)), "Expected calibration error of single models versus their seed ensemble")]
    out.append(axis(g, x, 0.0, 0.03 if g.narrow else 0.035, step, y0, y1,
                    lambda t: f"{t:.3f}".rstrip("0").rstrip(".") if t else "0",
                    "Expected calibration error (lower is better)"))
    for i, d in enumerate(cal):
        y = g.row_y(i)
        cls = "s1" if d["mixup"] else "s2"
        out.append(g.label(i, d["label"]))
        out.append(f'<line class="connector" x1="{x(d["member"]):.1f}" x2="{x(d["ensemble"]):.1f}" '
                   f'y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(dot(x(d["member"]), y, cls, hollow=True))
        out.append(dot(x(d["ensemble"]), y, cls))
        verdict = "worse" if d["ensemble"] > d["member"] else "better"
        out.append(hit(g, i, [f"ECE {d['member']:.4f} → {d['ensemble']:.4f} ({verdict})", d["label"],
                              "trained with Mixup/CutMix" if d["mixup"] else "trained without Mixup/CutMix",
                              f"accuracy {pct(d['acc_member'])} → {pct(d['acc_ens'])}"]))
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Tables (the accessible twin of every chart)
# --------------------------------------------------------------------------- #

def table(head: List[str], rows: List[List[str]], caption: str) -> str:
    th = "".join(f"<th>{E(h)}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{E(c)}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<details class="tableview"><summary>Show as table</summary>'
            f'<div class="scroll"><table><caption>{E(caption)}</caption><thead><tr>{th}</tr></thead>'
            f'<tbody>{body}</tbody></table></div></details>')


def legend(items: List[tuple]) -> str:
    out = ['<ul class="legend">']
    for cls, name, hollow in items:
        style = (f"background:var(--surface);box-shadow:inset 0 0 0 2.5px var(--{cls})" if hollow
                 else f"background:var(--{cls})")
        out.append(f'<li><span class="key" style="{style}"></span>{E(name)}</li>')
    out.append("</ul>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

def git_rev() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, cwd=ROOT, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


CSS = """
:root{
  color-scheme:light;
  --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6f6d68;
  --grid:#e1e0d9;--base:#c3c2b7;--ring:rgba(11,11,11,.10);--code:#f0efec;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--link:#1c5cab;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])){
    color-scheme:dark;
    --page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#9a988f;
    --grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);--code:#242422;
    --s1:#3987e5;--s2:#d95926;--s3:#199e70;--link:#86b6ef;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#9a988f;
  --grid:#2c2c2a;--base:#383835;--ring:rgba(255,255,255,.10);--code:#242422;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--link:#86b6ef;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--link)}
.wrap{max-width:880px;margin:0 auto;padding:0 16px}
header.top{padding:40px 0 8px}
.eyebrow{color:var(--ink-2);font-size:14px;margin:0 0 6px}
h1{font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0 0 12px;letter-spacing:-.01em}
.lede{font-size:18px;color:var(--ink-2);margin:0 0 20px;max-width:680px}
.links{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 8px;padding:0;list-style:none}
.links a,.theme{display:inline-block;padding:7px 14px;border-radius:999px;border:1px solid var(--ring);
  background:var(--surface);color:var(--ink);text-decoration:none;font-size:14px;cursor:pointer;font:inherit;font-size:14px}
.links a:hover,.theme:hover{border-color:var(--base)}
h2{font-size:24px;line-height:1.25;margin:48px 0 8px}
h3{font-size:18px;margin:0 0 4px}
p{margin:0 0 14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0 8px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px}
.tile .k{color:var(--ink-2);font-size:14px}
.tile .v{font-size:34px;font-weight:600;line-height:1.2;margin:4px 0}
.tile .d{color:var(--muted);font-size:13px}
ol.findings{padding-left:22px;margin:0}
ol.findings li{margin:0 0 12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:20px 20px 12px;margin:20px 0}
.card .sub{color:var(--ink-2);font-size:15px;margin:0 0 12px}
.chart{display:block;width:100%;height:auto;overflow:visible}
.chart.narrow{display:none}
@media (max-width:600px){.chart.wide{display:none}.chart.narrow{display:block}}
.chart text{fill:var(--ink-2);font-size:13px}
.chart .label,.chart .value,.chart .hint{paint-order:stroke;stroke:var(--surface);stroke-width:4px;stroke-linejoin:round}
.chart .label{fill:var(--ink);font-size:13.5px}
.chart .sublabel{fill:var(--muted);font-size:11px}
.chart .tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.chart .axis-title{fill:var(--muted);font-size:12px}
.chart .hint{fill:var(--muted);font-size:12px}
.chart .value{fill:var(--ink);font-size:12.5px;font-weight:600}
.chart .grid{stroke:var(--grid);stroke-width:1}
.chart .zero{stroke:var(--base);stroke-width:1.5}
.chart .whisker{stroke-width:2;stroke-linecap:round}
.chart .connector{stroke:var(--base);stroke-width:2;stroke-linecap:round}
.chart .hit{fill:transparent;cursor:default;outline:none}
.chart .hit:hover,.chart .hit:focus-visible{fill:var(--ink);fill-opacity:.04}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:0 0 10px;font-size:14px;color:var(--ink-2)}
.legend .key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:7px;vertical-align:0}
.note{color:var(--muted);font-size:13px;margin:8px 0 0}
details.tableview{margin:10px 0 4px}
details.tableview summary{cursor:pointer;color:var(--link);font-size:14px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;margin:10px 0}
caption{text-align:left;color:var(--ink-2);font-size:13px;padding-bottom:6px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
td{font-variant-numeric:tabular-nums}
pre{background:var(--code);border-radius:10px;padding:14px;overflow-x:auto;font-size:13px;line-height:1.5}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.callout{border-left:3px solid var(--base);padding:2px 0 2px 14px;margin:0 0 14px;color:var(--ink-2)}
footer{color:var(--muted);font-size:13px;padding:40px 0 48px;border-top:1px solid var(--grid);margin-top:56px}
#tip{position:fixed;pointer-events:none;z-index:10;background:var(--surface);color:var(--ink-2);
  border:1px solid var(--ring);border-radius:8px;padding:8px 10px;font-size:13px;line-height:1.45;
  box-shadow:0 4px 16px rgba(0,0,0,.12);max-width:300px;opacity:0;transition:opacity .08s}
#tip b{display:block;color:var(--ink);font-size:15px;font-weight:600}
@media (max-width:560px){.card{padding:14px 12px 8px}.tile .v{font-size:28px}}
"""

JS = """
(function(){
  var root=document.documentElement, btn=document.getElementById('theme');
  function current(){var t=root.getAttribute('data-theme');
    if(t)return t;return matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}
  function label(){btn.textContent=current()==='dark'?'Light theme':'Dark theme';}
  try{var saved=localStorage.getItem('fm-theme');if(saved)root.setAttribute('data-theme',saved);}catch(e){}
  label();
  btn.addEventListener('click',function(){var n=current()==='dark'?'light':'dark';
    root.setAttribute('data-theme',n);try{localStorage.setItem('fm-theme',n);}catch(e){}label();});

  var tip=document.getElementById('tip');
  function show(el,x,y){
    var lines=(el.getAttribute('data-tip')||'').split('\\n');
    tip.textContent='';
    var b=document.createElement('b');b.textContent=lines[0];tip.appendChild(b);
    for(var i=1;i<lines.length;i++){var d=document.createElement('div');d.textContent=lines[i];tip.appendChild(d);}
    var r=tip.getBoundingClientRect(), px=Math.min(x+14, innerWidth-r.width-8), py=y+14;
    if(py+r.height>innerHeight-8)py=y-r.height-14;
    tip.style.left=Math.max(8,px)+'px';tip.style.top=Math.max(8,py)+'px';tip.style.opacity=1;
  }
  function hide(){tip.style.opacity=0;}
  document.querySelectorAll('.chart .hit').forEach(function(el){
    var t=el.querySelector('title'); if(t) t.remove();
    el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});
    el.addEventListener('pointerleave',hide);
    el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.left+r.width*0.45,r.top+r.height/2);});
    el.addEventListener('blur',hide);
  });
})();
"""


def page(d) -> str:
    models, crop, pre, cal = d["models"], d["crop"], d["pretrain"], d["calib"]
    best, best_s = d["best"], d["best_scratch"]
    rev = git_rev()
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    vit = next(p for p in pre if p["label"].startswith("ViT"))
    cnn_gains = [p["pre"] - p["scr"] for p in pre if not p["label"].startswith("ViT")]
    small = [c for c in crop if c["params"] < 1e6]
    resnet_crop = next(c for c in crop if c["label"].startswith("ResNet"))
    mix = [c for c in cal if c["mixup"]]
    nomix = [c for c in cal if not c["mixup"]]

    models_tbl = table(
        ["Model", "Group", "Test accuracy", "95% CI (±)", "Seeds", "Params", "Note"],
        [[m["label"], CAT[m["cat"]][1], pct(m["mean"]), f"{m['ci'] * 100:.2f}" if m["ci"] else "–",
          str(m["n"]), f"{m['params'] / 1e6:.2f}M", m["note"]] for m in models],
        "Test accuracy on the official 10,000-image test set")
    crop_tbl = table(
        ["Model", "Params", "Δ from removing crop (points)", "95% CI", "p (paired t)", "Seeds better"],
        [[c["label"], f"{c['params'] / 1e6:.2f}M", pts(c["diff"]),
          f"[{pts(c['diff'] - c['ci'])}, {pts(c['diff'] + c['ci'])}]", f"{c['p']:.3f}",
          f"{c['wins']} / {c['n']}"] for c in crop],
        "Paired by seed against the same model trained with random crop")
    pre_tbl = table(
        ["Backbone", "From scratch", "Pretrained", "Δ (points)", "Params"],
        [[p["label"], pct(p["scr"]), pct(p["pre"]), pts(p["pre"] - p["scr"], 1),
          f"{p['params'] / 1e6:.1f}M"] for p in pre],
        "Mean of 3 seeds, pre-fix augmentation pipeline (each pair shares it)")
    cal_tbl = table(
        ["Group", "Mixup/CutMix", "Member ECE", "Ensemble ECE", "Member acc.", "Ensemble acc."],
        [[c["label"], "yes" if c["mixup"] else "no", f"{c['member']:.4f}", f"{c['ensemble']:.4f}",
          pct(c["acc_member"]), pct(c["acc_ens"])] for c in cal],
        "Five-seed ensembles, fixed augmentation pipeline")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fashion-MNIST Analysis</title>
<meta name="description" content="A reproducible multi-seed study of Fashion-MNIST: pretraining, augmentation, seed ensembles, calibration and robustness.">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%232a78d6'/%3E%3Ccircle cx='10' cy='20' r='3.5' fill='white'/%3E%3Ccircle cx='16' cy='13' r='3.5' fill='white'/%3E%3Ccircle cx='23' cy='9' r='3.5' fill='white'/%3E%3C/svg%3E">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <p class="eyebrow">Research project · PyTorch · {d["n_runs"]} training runs</p>
  <h1>Fashion-MNIST Analysis</h1>
  <p class="lede">What pretrained backbones, augmentation and seed ensembles actually buy on a small
  grayscale benchmark, and what they cost in calibration and robustness. Every number is a mean over
  several random seeds, with confidence intervals and paired tests.</p>
  <ul class="links">
    <li><a href="{REPO_URL}">Code on GitHub</a></li>
    <li><a href="{REPO_URL}#readme">Full write-up</a></li>
    <li><a href="{REPO_URL}/tree/main/results/sweeps">Raw results</a></li>
    <li><button class="theme" id="theme" type="button">Dark theme</button></li>
  </ul>
</header>

<section class="tiles" aria-label="Headline numbers">
  <div class="tile"><div class="k">Best test accuracy</div><div class="v">{pct(best["mean"])}</div>
    <div class="d">{E(best["label"])}, pretrained</div></div>
  <div class="tile"><div class="k">Best without pretraining</div><div class="v">{pct(best_s["mean"])}</div>
    <div class="d">{E(best_s["label"])}</div></div>
  <div class="tile"><div class="k">Pretraining gain, ViT</div><div class="v">{pts(vit["pre"] - vit["scr"], 1)}</div>
    <div class="d">points over the same model from scratch</div></div>
  <div class="tile"><div class="k">Training runs</div><div class="v">{d["n_runs"]}</div>
    <div class="d">3–5 seeds per configuration</div></div>
</section>

<h2>Main findings</h2>
<ol class="findings">
  <li><strong>Pretraining is what moves accuracy past about 94%.</strong> ImageNet weights improve every
  backbone on every seed: +{min(cnn_gains) * 100:.1f} to +{max(cnn_gains) * 100:.1f} points for CNNs and
  {pts(vit["pre"] - vit["scr"], 1)} for ViT-Tiny, which collapses to {pct(vit["scr"], 1)} from scratch.</li>
  <li><strong>Among models trained from scratch, ResNet-18 is clearly best</strong>
  ({pct(best_s["mean"])}), with a confidence interval that does not overlap TinyVGG's. The original
  single-seed claim that TinyVGG was best does not hold.</li>
  <li><strong>Whether random cropping helps depends on model size.</strong> Removing it gains
  {" and ".join(f"{pts(c['diff'], 1)} for {c['label']}" for c in small)} (about 0.1M parameters each),
  but costs ResNet-18 {abs(resnet_crop["diff"]) * 100:.1f} points, on every seed.</li>
  <li><strong>Seed ensembles add accuracy but can hurt calibration.</strong> Averaging five models worsens
  calibration for every group trained with Mixup/CutMix and improves it for those trained without,
  reproducing Wen et&nbsp;al. (ICLR&nbsp;2021).</li>
  <li><strong>Clean accuracy hides robustness differences.</strong> At near-identical accuracy, ResNet-18
  makes 16–17 points fewer errors than TinyVGG under contrast and brightness shifts.</li>
</ol>

<h2>How the models compare</h2>
<div class="card">
  <h3>Test accuracy by model</h3>
  <p class="sub">Dots are means over seeds; whiskers are 95% confidence intervals.</p>
  {legend([("s1", "Trained from scratch", False), ("s2", "ImageNet-pretrained", False), ("s3", "Seed ensemble", False)])}
  {both(chart_models, models)}
  <p class="note">† Measured on the pre-fix augmentation pipeline; the two best backbones were re-run after
  the fix and did not change. The best classical baseline, kNN on PCA features, reaches 85.8%.</p>
  {models_tbl}
</div>

<h2>Random cropping: it depends on model size</h2>
<p>The same augmentation that the two small CNNs are better off without is worth half a point to ResNet.
Fashion-MNIST items are already centred and size-normalised, so shifted crops mostly add noise for a
model with about 0.1M parameters, while a larger one can use them as regularisation.</p>
<div class="card">
  <h3>Accuracy change from removing random crop</h3>
  <p class="sub">Paired by seed against the same model with crop, ordered by parameter count.</p>
  {both(chart_crop, crop)}
  {crop_tbl}
</div>

<h2>Pretraining</h2>
<div class="card">
  <h3>From scratch versus ImageNet-pretrained</h3>
  <p class="sub">Each row is one backbone; the label is the gain in points.</p>
  {legend([("s1", "From scratch", False), ("s2", "Pretrained", False)])}
  {both(chart_pretrain, pre)}
  {pre_tbl}
</div>

<h2>Ensembles and calibration</h2>
<p>Training with Mixup or CutMix leaves single models slightly underconfident. Averaging several of them
compounds this, so the ensemble is worse calibrated than its members. Without Mixup/CutMix, single models are
overconfident and averaging corrects them.</p>
<div class="card">
  <h3>Calibration error of single models and their five-seed ensemble</h3>
  <p class="sub">Hollow dot: average single model. Filled dot: ensemble. Lower is better.</p>
  {legend([("s1", "Trained with Mixup/CutMix", False), ("s2", "Trained without Mixup/CutMix", False),
           ("muted", "Average single model", True), ("muted", "Five-seed ensemble", False)])}
  {both(chart_calib, cal)}
  <p class="note">With Mixup/CutMix, ECE rises from {min(c["member"] for c in mix):.3f}–{max(c["member"] for c in mix):.3f}
  to {min(c["ensemble"] for c in mix):.3f}–{max(c["ensemble"] for c in mix):.3f}; without it, ECE falls to
  {min(c["ensemble"] for c in nomix):.3f}–{max(c["ensemble"] for c in nomix):.3f}.</p>
  {cal_tbl}
</div>

<h2>Methods and corrections</h2>
<p>The official training images are split 48,000 / 12,000 into train and validation with a seeded split. The
official 10,000 test images are used once per run, on the best-validation checkpoint. Comparisons use paired
t-tests across shared seeds. Every run records its git commit, full configuration and per-epoch curves.</p>
<p class="callout"><strong>Two bugs were found and fixed during the study.</strong> The original accuracy
metric averaged per-batch accuracies, over-weighting the last partial batch. The augmentation pipeline applied
one random crop, flip and rotation to a whole batch and padded with grey instead of black. The fix was
measured against an exact reproduction of the old pipeline: it gains ResNet 1.6 points and changes one
conclusion, the ResNet versus TinyVGG ranking. Details are in the
<a href="{REPO_URL}#methods-and-corrections">write-up</a>.</p>

<h2>Reproduce</h2>
<pre><code>git clone {REPO_URL}.git
cd FashionMNIST-Analysis
pip install -r requirements.txt
python src/cli/prepare_data.py --output-dir data/processed

python src/cli/sweep.py expand sweeps/baseline_fixed.yaml        # 15 runs
python src/cli/sweep.py run    sweeps/baseline_fixed.yaml --all  # or as a SLURM job array
python src/cli/aggregate.py    runs/baseline_fixed --out results/sweeps/baseline_fixed</code></pre>

<footer>
  Built from commit <a href="{REPO_URL}/commit/{E(rev)}">{E(rev)}</a> on {built} by
  <code>site/build.py</code> from the CSV files in <code>results/sweeps/</code>.
  MIT licensed. Fashion-MNIST is provided by Zalando Research.
</footer>
</div>
<div id="tip" role="tooltip"></div>
<script>{JS}</script>
</body>
</html>
"""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "_site"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "index.html").write_text(page(load()), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    print(f"wrote {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
