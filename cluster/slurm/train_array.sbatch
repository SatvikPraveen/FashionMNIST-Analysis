#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM job array: one array task = one row of a sweep manifest.
#
#   python src/cli/sweep.py expand sweeps/baseline_seeds.yaml   # prints N
#   sbatch --array=0-$((N-1))%8 cluster/slurm/train_array.sbatch sweeps/baseline_seeds.yaml
#
# The %8 caps concurrent tasks (be a good neighbour). Each run resumes from its
# own <run>/<model>_last.pt, so `scontrol requeue` / preemption is safe.
# Edit the #SBATCH lines and the "site setup" block for your cluster.
# ---------------------------------------------------------------------------
#SBATCH --job-name=fmnist
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err
#SBATCH --partition=gpu            # <- site specific
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

SWEEP="${1:?usage: sbatch --array=0-N cluster/slurm/train_array.sbatch sweeps/<name>.yaml}"
shift || true

# ---- site setup (edit) ----------------------------------------------------
# module purge
# module load cuda/12.4 python/3.11
# source "$HOME/venvs/fmnist/bin/activate"
# conda activate fmnist

# Shared caches so compute nodes reuse what cluster/prefetch.py downloaded
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"
export HF_HUB_OFFLINE=1            # never try to reach the internet from a node
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
# ---------------------------------------------------------------------------

mkdir -p logs/slurm
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}[${SLURM_ARRAY_TASK_ID:-}] gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python src/cli/sweep.py run "$SWEEP" --index "${SLURM_ARRAY_TASK_ID}" \
    -- --num-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
