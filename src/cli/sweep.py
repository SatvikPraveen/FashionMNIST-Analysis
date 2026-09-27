#!/usr/bin/env python3
"""
Sweep runner: turn a sweep YAML into a manifest of independent training runs
and execute one (or all) of them.

Designed for SLURM job arrays: ``expand`` once on the login node, then each
array task runs ``run --index $SLURM_ARRAY_TASK_ID``. Every run is a normal
``train.py`` invocation, so anything the CLI accepts can be swept.

Sweep file format (see sweeps/*.yaml):

    name: baseline_seeds
    output_root: runs/baseline_seeds        # one sub-dir per run
    config: config.yaml                     # base config (optional)
    base_args: ["--amp", "--use-csv", ...]  # passed to every run (optional)
    models: [minicnn, tinyvgg, resnet]
    seeds: [0, 1, 2, 3, 4]
    variants:                               # named override sets (optional)
      default: {}
      no_mix: {augmentation.mixup: false, augmentation.cutmix: false}
    grid:                                   # cartesian product (optional)
      training.learning_rate: [1e-3, 3e-4]

Usage:
    python src/cli/sweep.py expand sweeps/baseline_seeds.yaml
    python src/cli/sweep.py run    sweeps/baseline_seeds.yaml --index 3
    python src/cli/sweep.py run    sweeps/baseline_seeds.yaml --all      # sequential
    python src/cli/sweep.py status sweeps/baseline_seeds.yaml
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _slug(value: Any) -> str:
    s = str(value).replace("-", "m") if isinstance(value, (int, float)) and value < 0 else str(value)
    return re.sub(r"[^A-Za-z0-9_.]+", "-", s).strip("-")


def load_sweep(path: str) -> Dict[str, Any]:
    with open(path) as f:
        sweep = yaml.safe_load(f) or {}
    sweep.setdefault("name", Path(path).stem)
    sweep.setdefault("output_root", f"runs/{sweep['name']}")
    sweep.setdefault("config", "config.yaml")
    sweep.setdefault("base_args", [])
    sweep.setdefault("seeds", [42])
    sweep.setdefault("variants", {"default": {}})
    sweep.setdefault("grid", {})
    if "models" not in sweep or not sweep["models"]:
        raise ValueError(f"{path}: 'models' must list at least one model")
    return sweep


def expand(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cartesian product of models x variants x grid x seeds -> run rows."""
    grid_keys = list(sweep["grid"].keys())
    grid_values = [sweep["grid"][k] for k in grid_keys]
    grid_combos = list(itertools.product(*grid_values)) if grid_keys else [()]

    rows: List[Dict[str, Any]] = []
    idx = 0
    for model in sweep["models"]:
        for variant, v_over in sweep["variants"].items():
            for combo in grid_combos:
                g_over = dict(zip(grid_keys, combo))
                overrides = {**(v_over or {}), **g_over}
                grid_tag = "_".join(f"{k.split('.')[-1]}{_slug(v)}" for k, v in g_over.items())
                group = "|".join(x for x in (model, variant, grid_tag) if x)
                for seed in sweep["seeds"]:
                    run_name = "_".join(x for x in (model, variant if variant != "default" else "",
                                                    grid_tag, f"seed{seed}") if x)
                    rows.append({
                        "index": idx,
                        "run_name": run_name,
                        "group": group,
                        "model": model,
                        "variant": variant,
                        "seed": int(seed),
                        "overrides": overrides,
                        "output_dir": os.path.join(sweep["output_root"], run_name),
                    })
                    idx += 1
    return rows


def manifest_paths(sweep: Dict[str, Any]):
    root = Path(sweep["output_root"])
    return root / "manifest.jsonl", root / "manifest.csv"


def write_manifest(sweep: Dict[str, Any], rows: List[Dict[str, Any]]) -> Path:
    jsonl, csv_path = manifest_paths(sweep)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "run_name", "group", "model", "variant", "seed", "overrides", "output_dir"])
        for r in rows:
            w.writerow([r["index"], r["run_name"], r["group"], r["model"], r["variant"],
                        r["seed"], json.dumps(r["overrides"]), r["output_dir"]])
    with open(jsonl.parent / "sweep.yaml", "w") as f:
        yaml.safe_dump(sweep, f, sort_keys=False)
    return jsonl


def read_manifest(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    jsonl, _ = manifest_paths(sweep)
    if not jsonl.exists():
        return expand(sweep)
    with open(jsonl) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def build_argv(sweep: Dict[str, Any], row: Dict[str, Any], extra: Optional[List[str]] = None) -> List[str]:
    argv = [
        "--config", sweep["config"],
        "--model", row["model"],
        "--seed", str(row["seed"]),
        "--output-dir", row["output_dir"],
        "--run-name", row["run_name"],
        "--skip-best-selection",
        "--resume",                          # safe on requeue / preemption
        "--set", f"sweep.name={sweep['name']}",
        "--set", f"sweep.group={row['group']}",
        "--set", f"sweep.variant={row['variant']}",
        "--set", f"sweep.run_name={row['run_name']}",
    ]
    argv += list(sweep["base_args"])
    for k, v in row["overrides"].items():
        argv += ["--set", f"{k}={json.dumps(v)}"]
    argv += list(extra or [])
    return argv


def run_row(sweep: Dict[str, Any], row: Dict[str, Any], extra: Optional[List[str]] = None,
            dry_run: bool = False) -> None:
    argv = build_argv(sweep, row, extra)
    cmd = "python src/cli/train.py " + " ".join(_quote(a) for a in argv)
    print(f"[sweep {sweep['name']}] run #{row['index']} {row['run_name']}\n  {cmd}", flush=True)
    if dry_run:
        return
    from src.training.trainer import main as train_main
    train_main(argv)


def _quote(a: str) -> str:
    import shlex
    return shlex.quote(a)


def run_status(row: Dict[str, Any]) -> str:
    run_json = Path(row["output_dir"]) / row["model"] / "run.json"
    if not run_json.exists():
        return "pending"
    try:
        with open(run_json) as f:
            return json.load(f).get("status", "running")
    except json.JSONDecodeError:
        return "running"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("expand", help="write manifest.jsonl/csv under output_root")
    pe.add_argument("sweep")

    pr = sub.add_parser("run", help="run one manifest row (or all sequentially)")
    pr.add_argument("sweep")
    g = pr.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="row index (e.g. $SLURM_ARRAY_TASK_ID)")
    g.add_argument("--all", action="store_true", help="run every row sequentially")
    pr.add_argument("--only-pending", action="store_true", help="with --all: skip finished rows")
    pr.add_argument("--dry-run", action="store_true", help="print commands only")
    pr.add_argument("extra", nargs="*", help="extra args after '--' passed to train.py")

    ps = sub.add_parser("status", help="show finished / failed / pending rows")
    ps.add_argument("sweep")

    args = p.parse_args(argv)
    sweep = load_sweep(args.sweep)

    if args.cmd == "expand":
        rows = expand(sweep)
        path = write_manifest(sweep, rows)
        print(f"{len(rows)} runs -> {path}")
        print(f"SLURM array range: 0-{len(rows) - 1}")
        return 0

    rows = read_manifest(sweep)

    if args.cmd == "status":
        counts: Dict[str, int] = {}
        for r in rows:
            st = run_status(r)
            counts[st] = counts.get(st, 0) + 1
            print(f"{r['index']:4d}  {st:9s}  {r['run_name']}")
        print("\n" + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        return 0

    if args.cmd == "run":
        if args.all:
            todo = [r for r in rows if not (args.only_pending and run_status(r) == "finished")]
        else:
            if not 0 <= args.index < len(rows):
                raise SystemExit(f"--index {args.index} out of range 0-{len(rows) - 1}")
            todo = [rows[args.index]]
        for r in todo:
            run_row(sweep, r, args.extra, dry_run=args.dry_run)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
