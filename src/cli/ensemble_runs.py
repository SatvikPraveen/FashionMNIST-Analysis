#!/usr/bin/env python3
"""
Seed ensembles per sweep group: average the softmax of every finished
checkpoint in a group on the official test set, and compare the ensemble
with the average single member.

Deep ensembles of independently seeded networks are the standard, cheap
baseline for both accuracy and calibration gains; every sweep already has
several seeds per group, so this only costs inference.

Usage:
    python src/cli/ensemble_runs.py runs/baseline_seeds --out results/sweeps/baseline_seeds
    python src/cli/ensemble_runs.py runs/backbones --groups 'resnet18|pretrained'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402

from src.evaluation.analysis import (  # noqa: E402
    collect_outputs, expected_calibration_error, negative_log_likelihood, brier_score,
)
from src.models.registry import load_model_from_checkpoint  # noqa: E402
from src.cli.analyze_runs import find_finished_runs  # noqa: E402


def group_runs(run_dirs: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for d in run_dirs:
        run = json.loads((d / "run.json").read_text())
        g = ((run.get("config", {}) or {}).get("sweep", {}) or {}).get("group") \
            or (run.get("params", {}) or {}).get("model")
        groups.setdefault(g, []).append(d)
    return groups


def member_probs(run_dir: Path, test_loader, device: torch.device) -> torch.Tensor:
    run = json.loads((run_dir / "run.json").read_text())
    name = run["params"]["model"]
    model = load_model_from_checkpoint(str(run_dir / f"{name}_best.pth"),
                                       map_location=str(device)).to(device)
    logits, _ = collect_outputs(model, test_loader, device)
    return logits.softmax(dim=1)


def _metrics(probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logp = probs.clamp_min(1e-12).log()
    return {
        "accuracy": float((probs.argmax(1) == labels).float().mean()),
        "nll": negative_log_likelihood(logp, labels),   # CE on log-probs == NLL
        "brier": brier_score(probs, labels),
        "ece": expected_calibration_error(probs, labels)["ece"],
    }


def ensemble_group(prob_list: List[torch.Tensor], labels: torch.Tensor) -> Dict[str, Any]:
    members = [_metrics(p, labels) for p in prob_list]
    ens = _metrics(torch.stack(prob_list).mean(0), labels)
    n = len(members)
    out: Dict[str, Any] = {"n_members": n}
    for k in ("accuracy", "nll", "brier", "ece"):
        out[f"member_{k}_mean"] = sum(m[k] for m in members) / n
        out[f"ensemble_{k}"] = ens[k]
    out["member_accuracy_best"] = max(m["accuracy"] for m in members)
    out["accuracy_gain"] = out["ensemble_accuracy"] - out["member_accuracy_mean"]
    # How often do members disagree? (fraction of test images without a unanimous vote)
    preds = torch.stack([p.argmax(1) for p in prob_list])
    out["disagreement"] = float((preds != preds[0]).any(0).float().mean())
    return out


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_no groups_\n"
    lines = ["| group | members | mean member acc | best member | **ensemble acc** | gain "
             "| member ECE → ensemble ECE | member NLL → ensemble NLL | disagreement |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['n_members']} | {r['member_accuracy_mean']:.4f} "
            f"| {r['member_accuracy_best']:.4f} | **{r['ensemble_accuracy']:.4f}** "
            f"| {r['accuracy_gain']:+.4f} | {r['member_ece_mean']:.4f} → {r['ensemble_ece']:.4f} "
            f"| {r['member_nll_mean']:.4f} → {r['ensemble_nll']:.4f} | {r['disagreement']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", default=None, help="output prefix; writes <out>_ensemble.{csv,md}")
    p.add_argument("--groups", nargs="*", default=None, help="only these groups")
    p.add_argument("--min-members", type=int, default=2)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args(argv)

    from src.data.dataset import create_dataloaders
    from src.training.utils import get_device
    device = get_device()
    # The official test set does not depend on the seed; any seed works here.
    _, _, test_loader = create_dataloaders(use_torchvision=True, batch_size=args.batch_size,
                                           num_workers=args.num_workers, seed=0,
                                           data_root=args.data_root)
    labels = torch.cat([y for _, y in test_loader])

    groups = group_runs(find_finished_runs(args.roots))
    rows = []
    for g in sorted(groups):
        if args.groups and g not in args.groups:
            continue
        if len(groups[g]) < args.min_members:
            continue
        probs = [member_probs(d, test_loader, device) for d in groups[g]]
        row = {"group": g, **ensemble_group(probs, labels)}
        rows.append(row)
        print(f"{g}: {row['n_members']} members, ensemble acc {row['ensemble_accuracy']:.4f} "
              f"(mean member {row['member_accuracy_mean']:.4f})", flush=True)
    rows.sort(key=lambda r: -r["ensemble_accuracy"])
    print("\n" + to_markdown(rows))
    if args.out:
        import pandas as pd
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(f"{args.out}_ensemble.csv", index=False)
        Path(f"{args.out}_ensemble.md").write_text(to_markdown(rows))
        print(f"wrote {args.out}_ensemble.csv, {args.out}_ensemble.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
