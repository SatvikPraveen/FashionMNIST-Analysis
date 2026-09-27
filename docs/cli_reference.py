#!/usr/bin/env python3
"""
Regenerate docs/CLI_REFERENCE.md from the real --help output of every
command-line tool, so the reference cannot drift from the code.

    python docs/cli_reference.py

tests/test_cli_reference.py fails when a flag exists in a tool but not in
the committed reference, which is the cue to re-run this script.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "CLI_REFERENCE.md"

# (title, argv relative to the repo root, one-line purpose)
TOOLS = [
    ("prepare_data.py", ["src/cli/prepare_data.py"], "Download Fashion-MNIST and write seeded train/val/test CSV splits."),
    ("train.py", ["src/cli/train.py"], "Train one or more models; every run writes run.json, metrics.jsonl and checkpoints."),
    ("evaluate.py", ["src/cli/evaluate.py"], "Evaluate one checkpoint: metrics, confusion matrix and, with --analysis, calibration and robustness."),
    ("finetune.py", ["src/cli/finetune.py"], "Grid search over learning rate, batch size and patience (sequential, one machine)."),
    ("sweep.py", ["src/cli/sweep.py"], "Expand a sweep YAML into runs, run them (locally or as a job array) and show progress."),
    ("sweep.py expand", ["src/cli/sweep.py", "expand"], "Write the run manifest for a sweep."),
    ("sweep.py run", ["src/cli/sweep.py", "run"], "Run one manifest row (--index) or all rows (--all). Arguments after -- go to train.py."),
    ("sweep.py status", ["src/cli/sweep.py", "status"], "List finished, failed and pending runs."),
    ("aggregate.py", ["src/cli/aggregate.py"], "Summarise runs: mean, std, 95% CI per group; --baseline adds paired-by-seed tests."),
    ("analyze_runs.py", ["src/cli/analyze_runs.py"], "Calibration, per-class and robustness analysis for every checkpoint in a sweep."),
    ("ensemble_runs.py", ["src/cli/ensemble_runs.py"], "Seed ensembles per group, compared with their members."),
]


def help_text(argv):
    env = dict(os.environ, COLUMNS="88", PYTHONWARNINGS="ignore")
    out = subprocess.run([sys.executable, *argv, "--help"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    lines = [l for l in out.stdout.splitlines()
             if " - INFO - " not in l and " - WARNING - " not in l]
    return "\n".join(lines).strip()


def main() -> int:
    parts = ["# Command-line reference", "",
             "Generated from each tool's `--help` by `python docs/cli_reference.py`; do not edit by hand.",
             "Task-oriented examples are in the [usage guide](USAGE_GUIDE.md).", ""]
    for title, argv, purpose in TOOLS:
        parts += [f"## {title}", "", purpose, "", "```text", help_text(argv), "```", ""]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
