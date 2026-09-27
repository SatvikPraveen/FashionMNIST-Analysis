#!/bin/bash
# ---------------------------------------------------------------------------
# Single training job (not a sweep). Everything after the script name is
# passed to src/cli/train.py, e.g.
#
#   sbatch cluster/slurm/train_single.sbatch --model resnet18 --pretrained --amp --epochs 30
#   sbatch cluster/slurm/train_single.sbatch --model all --amp --num-workers 8
# ---------------------------------------------------------------------------
#SBATCH --job-name=fmnist-single
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --partition=gpu            # <- site specific
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue

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

python src/cli/train.py --resume --num-workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
