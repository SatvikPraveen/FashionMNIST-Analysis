#!/usr/bin/env python3
"""
Batch calibration / per-class / robustness analysis over a sweep's checkpoints.

For every finished run under the given roots this rebuilds the model from its
``*_best_spec.json``, recreates *that run's* validation split from its seed
(for temperature scaling) and the official test set, runs
:func:`src.evaluation.analysis.run_full_analysis`, and writes
``<run_dir>/analysis/analysis.json`` plus figures. It then aggregates the
headline metrics by sweep group (mean ± std over seeds).

Usage:
    python src/cli/analyze_runs.py runs/baseline_seeds --out results/sweeps/baseline_seeds
    python src/cli/analyze_runs.py runs/a runs/b --out results/sweeps/ab --no-robustness
    sbatch cluster/slurm/python.sbatch src/cli/analyze_runs.py runs/baseline_seeds --out ...

Outputs with --out PREFIX: PREFIX_analysis_runs.csv (one row per run) and
PREFIX_analysis_summary.{csv,md} (one row per group).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402

from src.evaluation.analysis import run_full_analysis  # noqa: E402
from src.models.registry import load_model_from_checkpoint  # noqa: E402

LoaderFn = Callable[[int], Tuple[Any, Any]]

SUMMARY_METRICS = ["accuracy", "macro_f1", "nll", "brier", "ece", "mce", "temperature",
                   "nll_ts", "ece_ts", "mean_corruption_error", "relative_mce"]


def find_finished_runs(roots: List[str]) -> List[Path]:
    out = []
    for root in roots:
        for run_json in sorted(Path(root).rglob("run.json")):
            try:
                run = json.loads(run_json.read_text())
            except json.JSONDecodeError:
                continue
            if run.get("status") == "finished":
                out.append(run_json.parent)
    return out


def default_loaders(data_root: str, batch_size: int, num_workers: int) -> LoaderFn:
    """Return seed -> (val_loader, test_loader), matching train.py's split exactly."""
    from src.data.dataset import create_dataloaders
    cache: Dict[int, Tuple[Any, Any]] = {}

    def fn(seed: int):
        if seed not in cache:
            _, val, test = create_dataloaders(use_torchvision=True, batch_size=batch_size,
                                              num_workers=num_workers, seed=seed,
                                              data_root=data_root)
            cache.clear()  # keep one split in memory at a time
            cache[seed] = (val, test)
        return cache[seed]
    return fn


def analyze_run(run_dir: Path, loaders: LoaderFn, device: torch.device,
                robustness: bool = True, skip_existing: bool = True) -> Dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    params = run.get("params", {}) or {}
    sweep = (run.get("config", {}) or {}).get("sweep", {}) or {}
    model_name = params.get("model")
    seed = int(params.get("seed", 42))
    out_dir = run_dir / "analysis"
    report_path = out_dir / "analysis.json"

    if skip_existing and report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        weights = run_dir / f"{model_name}_best.pth"
        model = load_model_from_checkpoint(str(weights), map_location=str(device)).to(device)
        val_loader, test_loader = loaders(seed)
        report = run_full_analysis(model, test_loader, device, out_dir=str(out_dir),
                                   val_loader=val_loader, robustness=robustness)

    cal = report["calibration"]
    row: Dict[str, Any] = {
        "run_dir": str(run_dir), "group": sweep.get("group") or model_name,
        "variant": sweep.get("variant"), "model": model_name, "seed": seed,
        "accuracy": report["accuracy"], "macro_f1": report["per_class"]["macro_f1"],
        "nll": cal["nll"], "brier": cal["brier"], "ece": cal["ece"], "mce": cal["mce"],
        "temperature": cal.get("temperature"),
        "nll_ts": (cal.get("scaled") or {}).get("nll"),
        "ece_ts": (cal.get("scaled") or {}).get("ece"),
    }
    rob = report.get("robustness")
    if rob:
        row["mean_corruption_error"] = rob["mean_corruption_error"]
        row["relative_mce"] = rob["relative_mce"]
        for name, err in rob["mean_error_per_corruption"].items():
            row[f"err_{name}"] = err
    top = report["per_class"]["top_confusions"][:1]
    if top:
        row["top_confusion"] = f"{top[0]['true']}->{top[0]['pred']}"
    return row


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    out = []
    for g, members in groups.items():
        rec: Dict[str, Any] = {"group": g, "n": len(members)}
        for m in SUMMARY_METRICS:
            vals = [float(x[m]) for x in members if x.get(m) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                   if len(vals) > 1 else 0.0)
            rec[f"{m}_mean"], rec[f"{m}_std"] = mean, std
        out.append(rec)
    out.sort(key=lambda d: -d.get("accuracy_mean", 0))
    return out


def to_markdown(summary: List[Dict[str, Any]]) -> str:
    if not summary:
        return "_no analysed runs_\n"
    has_rob = any("mean_corruption_error_mean" in s for s in summary)

    def f(s, m, digits=4):
        if f"{m}_mean" not in s:
            return "—"
        return f"{s[f'{m}_mean']:.{digits}f} ± {s[f'{m}_std']:.{digits}f}"

    head = "| group | n | accuracy | ECE | ECE after T | T | NLL | NLL after T |"
    sep = "|---|---|---|---|---|---|---|---|"
    if has_rob:
        head += " mean corruption error |"
        sep += "---|"
    lines = [head, sep]
    for s in summary:
        line = (f"| {s['group']} | {s['n']} | {f(s, 'accuracy')} | {f(s, 'ece')} | {f(s, 'ece_ts')} "
                f"| {f(s, 'temperature', 3)} | {f(s, 'nll')} | {f(s, 'nll_ts')} |")
        if has_rob:
            line += f" {f(s, 'mean_corruption_error')} |"
        lines.append(line)
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", default=None, help="output prefix, e.g. results/sweeps/baseline_seeds")
    p.add_argument("--no-robustness", action="store_true")
    p.add_argument("--recompute", action="store_true", help="ignore existing analysis.json files")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args(argv)

    from src.training.utils import get_device
    device = get_device()
    runs = find_finished_runs(args.roots)
    print(f"{len(runs)} finished runs")
    loaders = default_loaders(args.data_root, args.batch_size, args.num_workers)
    # Group by seed so each train/val split is built once
    runs.sort(key=lambda d: json.loads((d / "run.json").read_text()).get("params", {}).get("seed", 0))

    rows = []
    for i, run_dir in enumerate(runs, 1):
        row = analyze_run(run_dir, loaders, device, robustness=not args.no_robustness,
                          skip_existing=not args.recompute)
        rows.append(row)
        print(f"[{i}/{len(runs)}] {row['group']} seed={row['seed']} acc={row['accuracy']:.4f} "
              f"ece={row['ece']:.4f} mCE={row.get('mean_corruption_error', float('nan')):.4f}", flush=True)

    summary = summarize(rows)
    print("\n" + to_markdown(summary))
    if args.out:
        import pandas as pd
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{args.out}_analysis_runs.csv", index=False)
        pd.DataFrame(summary).to_csv(f"{args.out}_analysis_summary.csv", index=False)
        Path(f"{args.out}_analysis_summary.md").write_text(to_markdown(summary))
        print(f"wrote {args.out}_analysis_runs.csv, _analysis_summary.csv, _analysis_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
