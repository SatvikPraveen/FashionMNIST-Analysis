# Running on a cluster (SLURM)

Everything below assumes a SLURM cluster with GPU nodes that may not have
internet access. Only `cluster/slurm/*.sbatch` are site-specific; the Python
code is identical on a laptop and on a node.

## 1. One-time setup (login node)

```bash
git clone https://github.com/SatvikPraveen/FashionMNIST-Analysis.git
cd FashionMNIST-Analysis

# environment (pick one)
python -m venv ~/venvs/fmnist && source ~/venvs/fmnist/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the cluster's CUDA
pip install -r cluster/requirements-cluster.txt

# caches on the shared filesystem so compute nodes can read them
export HF_HOME=$PWD/.cache/huggingface TORCH_HOME=$PWD/.cache/torch
python cluster/prefetch.py            # dataset + resnet18/efficientnet_b0/convnext_tiny/vit_tiny weights

# sanity check
pytest tests/ -q
```

Then open `cluster/slurm/train_array.sbatch` and fill in the *site setup*
block (module loads / venv activation / partition name).

## 2. Run a sweep as a job array

```bash
python src/cli/sweep.py expand sweeps/baseline_seeds.yaml
#   15 runs -> runs/baseline_seeds/manifest.jsonl
#   SLURM array range: 0-14

sbatch --array=0-14%8 cluster/slurm/train_array.sbatch sweeps/baseline_seeds.yaml
```

* Each array task runs exactly one manifest row (model × variant × grid × seed).
* Every run writes `runs/<sweep>/<run>/<model>/{run.json,metrics.jsonl,*_best.pth,*_last.pt}`.
* Runs are started with `--resume`, so a requeued / pre-empted task continues
  from its last completed epoch instead of restarting.
* `python src/cli/sweep.py status sweeps/baseline_seeds.yaml` shows
  finished / failed / pending rows; re-submit only the failed indices with
  `--array=3,7,11`.

Available sweeps:

| file | what it measures | runs |
|---|---|---|
| `sweeps/smoke.yaml` | 1-epoch check of the machinery | 2 |
| `sweeps/baseline_seeds.yaml` | 3 custom CNNs × 5 seeds | 15 |
| `sweeps/backbones.yaml` | 4 timm backbones, pretrained vs scratch × 3 seeds | 24 |
| `sweeps/augmentation_ablation.yaml` | remove one augmentation at a time × 5 seeds | 30 |
| `sweeps/lr_grid.yaml` | LR × weight-decay grid × 3 seeds | 18 |

Write your own by copying one; any `config.yaml` key can go under
`variants:` or `grid:`, and any `train.py` flag under `base_args:`.

## 3. Aggregate

```bash
python src/cli/aggregate.py runs/baseline_seeds --out results/baseline_seeds
```

prints and writes a table with `n`, mean ± std, 95% CI (t-distribution),
min/max, parameter count and time per run for every group. Several sweep
roots can be pooled: `aggregate.py runs/a runs/b --group-by model`.

## 4. Single jobs

```bash
sbatch cluster/slurm/train_single.sbatch --model resnet18 --pretrained --amp --epochs 30
```

## Notes

* **Throughput flags**: `--amp` (bf16 on A100/H100, fp16 elsewhere),
  `--compile` (torch.compile, CUDA only), `--num-workers` (set from
  `$SLURM_CPUS_PER_TASK` by the sbatch scripts).
* **Determinism**: `--deterministic` forces deterministic kernels for
  bit-exact reproduction at some speed cost; `--seed` alone is enough for
  statistically reproducible multi-seed studies.
* **Tracking**: `run.json` records git commit, hostname, SLURM job/array ids,
  GPU name, torch version and the full config. Add `--mlflow` or `--wandb`
  (and `pip install mlflow`/`wandb`) to mirror metrics to a server.
* **Offline nodes**: `HF_HUB_OFFLINE=1` is set in the sbatch scripts, so a
  backbone whose weights were not prefetched fails fast instead of hanging.
