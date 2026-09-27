#!/bin/bash
# Summarise a finished sweep (CPU-only, seconds).
#   sbatch cluster/slurm/aggregate.sbatch runs/baseline_seeds results/baseline_seeds
#SBATCH --job-name=fmnist-agg
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
ROOT="${1:?sweep output root}"; OUT="${2:-results/$(basename "$ROOT")}"
# source "$HOME/venvs/fmnist/bin/activate"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
python src/cli/aggregate.py "$ROOT" --out "$OUT"
