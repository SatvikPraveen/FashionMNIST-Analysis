#!/usr/bin/env python3
"""
Aggregate the run.json files of a sweep into a results table with mean,
standard deviation and a 95% confidence interval over seeds.

Usage:
    python src/cli/aggregate.py runs/baseline_seeds
    python src/cli/aggregate.py runs/baseline_seeds --metric test_acc --out results/baseline
    python src/cli/aggregate.py runs/a runs/b --group-by model   # pool several sweeps

Outputs (with --out PREFIX): PREFIX_runs.csv (one row per run) and
PREFIX_summary.csv + PREFIX_summary.md (one row per group).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def find_runs(roots: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for root in roots:
        for run_json in sorted(Path(root).rglob("run.json")):
            try:
                with open(run_json) as f:
                    run = json.load(f)
            except json.JSONDecodeError:
                continue
            params = run.get("params", {}) or {}
            summary = run.get("summary", {}) or {}
            cfg = run.get("config", {}) or {}
            sweep = cfg.get("sweep", {}) or {}
            rows.append({
                "run_dir": str(run_json.parent),
                "run_name": run.get("run_name"),
                "sweep": sweep.get("name"),
                "group": sweep.get("group") or params.get("model"),
                "variant": sweep.get("variant"),
                "model": params.get("model"),
                "seed": params.get("seed"),
                "status": run.get("status"),
                "test_acc": summary.get("test_acc"),
                "test_loss": summary.get("test_loss"),
                "best_val_acc": summary.get("best_val_acc"),
                "epochs_trained": summary.get("epochs_trained"),
                "train_time_sec": summary.get("train_time_sec"),
                "num_parameters": params.get("num_parameters"),
                "amp": params.get("amp"),
                "git_commit": (run.get("git") or {}).get("commit"),
                "slurm_job": (run.get("slurm") or {}).get("slurm_array_job_id")
                              or (run.get("slurm") or {}).get("slurm_job_id"),
            })
    return rows


# t critical values (two-sided 95%) for small n; falls back to 1.96
_T95 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201,
        13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
        19: 2.101, 20: 2.093, 21: 2.086, 25: 2.064, 30: 2.045}


def _t95(n: int) -> float:
    if n in _T95:
        return _T95[n]
    if n < 2:
        return float("nan")
    return 1.96 if n > 30 else _T95[max(k for k in _T95 if k <= n)]


def summarize(rows: List[Dict[str, Any]], metric: str = "test_acc",
              group_by: str = "group") -> List[Dict[str, Any]]:
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "finished" or r.get(metric) is None:
            continue
        groups.setdefault(r.get(group_by), []).append(r)

    out = []
    for key, members in groups.items():
        vals = [float(m[metric]) for m in members]
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        ci = _t95(n) * std / math.sqrt(n) if n > 1 else float("nan")
        times = [m["train_time_sec"] for m in members if m.get("train_time_sec") is not None]
        out.append({
            group_by: key,
            "model": members[0].get("model"),
            "variant": members[0].get("variant"),
            "n": n,
            f"{metric}_mean": mean,
            f"{metric}_std": std,
            f"{metric}_ci95": ci,
            f"{metric}_min": min(vals),
            f"{metric}_max": max(vals),
            "seeds": sorted(m.get("seed") for m in members if m.get("seed") is not None),
            "mean_train_time_sec": (sum(times) / len(times)) if times else None,
            "num_parameters": members[0].get("num_parameters"),
        })
    out.sort(key=lambda d: -d[f"{metric}_mean"])
    return out


def to_markdown(summary: List[Dict[str, Any]], metric: str, group_by: str) -> str:
    if not summary:
        return "_no finished runs_\n"
    lines = [f"| {group_by} | n | {metric} mean ± std | 95% CI | min | max | params | time/run |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        ci = s[f"{metric}_ci95"]
        ci_s = f"±{ci:.4f}" if ci == ci else "—"
        params = f"{s['num_parameters'] / 1e6:.2f}M" if s.get("num_parameters") else "—"
        t = f"{s['mean_train_time_sec'] / 60:.1f} min" if s.get("mean_train_time_sec") else "—"
        lines.append(f"| {s[group_by]} | {s['n']} | {s[f'{metric}_mean']:.4f} ± {s[f'{metric}_std']:.4f} "
                     f"| {ci_s} | {s[f'{metric}_min']:.4f} | {s[f'{metric}_max']:.4f} | {params} | {t} |")
    return "\n".join(lines) + "\n"


def write_outputs(rows, summary, prefix: str, metric: str, group_by: str) -> None:
    import pandas as pd
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{prefix}_runs.csv", index=False)
    pd.DataFrame(summary).to_csv(f"{prefix}_summary.csv", index=False)
    with open(f"{prefix}_summary.md", "w") as f:
        f.write(to_markdown(summary, metric, group_by))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="sweep output roots (searched recursively for run.json)")
    p.add_argument("--metric", default="test_acc")
    p.add_argument("--group-by", default="group", choices=["group", "model", "variant", "sweep"])
    p.add_argument("--out", default=None, help="output prefix, e.g. results/baseline")
    args = p.parse_args(argv)

    rows = find_runs(args.roots)
    summary = summarize(rows, args.metric, args.group_by)
    statuses: Dict[str, int] = {}
    for r in rows:
        statuses[r["status"] or "unknown"] = statuses.get(r["status"] or "unknown", 0) + 1
    print(f"{len(rows)} runs found ({', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))})\n")
    print(to_markdown(summary, args.metric, args.group_by))
    if args.out:
        write_outputs(rows, summary, args.out, args.metric, args.group_by)
        print(f"wrote {args.out}_runs.csv, {args.out}_summary.csv, {args.out}_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
