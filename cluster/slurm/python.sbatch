#!/bin/bash
# ---------------------------------------------------------------------------
# Run any project Python script on one GPU (analysis, ensembles, ...), e.g.
#
#   sbatch cluster/slurm/python.sbatch src/cli/analyze_runs.py runs/baseline_seeds --out results/sweeps/baseline_seeds
#   sbatch cluster/slurm/python.sbatch src/cli/ensemble_runs.py runs/baseline_seeds --out results/sweeps/seed_ensembles
# ---------------------------------------------------------------------------
#SBATCH --job-name=fmnist-py
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu            # <- site specific
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

# ---- site setup (edit) ----------------------------------------------------
# module purge
# module load cuda/12.4 python/3.11
# source "$HOME/venvs/fmnist/bin/activate"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# ---------------------------------------------------------------------------

mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR:-$PWD}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python "$@"
