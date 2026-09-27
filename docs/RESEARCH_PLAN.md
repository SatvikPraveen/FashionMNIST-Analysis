# Research plan: what does a modern training recipe buy on a small grayscale benchmark?

**Status:** infrastructure complete (branch `research-grade`, 2026-09-26); experiments not yet run.

Fashion-MNIST is small enough to run hundreds of controlled experiments and
well-studied enough that the literature gives clear reference points
(~94–95% for a ResNet-18-class CNN with standard augmentation, ~96–97% with
heavy augmentation or ensembles). That makes it a good testbed for an
*empirical* question rather than a leaderboard chase:

> How much of the gap between a small custom CNN and the state of the art is
> explained by (a) architecture / pretraining, (b) the augmentation recipe,
> (c) the optimisation recipe — and what does each choice cost in calibration
> and robustness?

## Research questions and the sweep that answers each

| # | Question | Sweep | Runs | Primary metric |
|---|---|---|---|---|
| Q1 | How much seed variance do the existing baselines have? Are the reported differences between MiniCNN / TinyVGG / ResNet real? | `sweeps/baseline_seeds.yaml` | 15 | test acc mean ± std, 95% CI |
| Q2 | Do ImageNet-pretrained backbones help on 28 px grayscale data, and does the answer differ for CNNs vs ViTs? | `sweeps/backbones.yaml` | 24 | test acc, params, time/run |
| Q3 | Which augmentation components matter? | `sweeps/augmentation_ablation.yaml` | 30 | Δ test acc vs full recipe |
| Q4 | Is the default LR / weight decay near-optimal for the small CNNs? | `sweeps/lr_grid.yaml` | 18 | test acc |
| Q5 | Are the more accurate models also better calibrated and more robust? | `evaluate.py --analysis` on each Q1/Q2 winner | – | ECE, NLL, mean corruption error |

Every sweep varies **seeds** (3–5) so every comparison comes with a
confidence interval. All numbers are on the **official 10,000-image test
set**, evaluated **once** from the best-validation checkpoint.

## Protocol

1. **Split.** Official train (60k) → 48k train / 12k val with a fixed seeded
   split; official test (10k) untouched. `src/cli/prepare_data.py` writes the
   CSVs; `create_dataloaders(seed=...)` reproduces the split.
2. **Seeds.** `set_seed` covers Python, NumPy, torch CPU/CUDA/MPS, DataLoader
   workers and the train/val split. Seeds 0–4.
3. **Model selection.** Early stopping on val accuracy; test evaluated once.
4. **Reporting.** `src/cli/aggregate.py` → mean, std, t-based 95% CI, min/max,
   parameters, time per run. Differences are called significant only when
   the CIs do not overlap or a paired-by-seed test says so.
5. **Provenance.** Every run's `run.json` holds git commit, config snapshot,
   argv, host, SLURM ids and device; `metrics.jsonl` holds per-epoch curves.
6. **Beyond accuracy.** For the winners: reliability diagram + ECE before and
   after temperature scaling; per-class F1 and the confusion pairs
   (T-shirt / Shirt / Pullover / Coat); accuracy under seven corruptions ×
   five severities.

## Compute plan

| sweep | runs | est. GPU-min / run (A100, `--amp`) | total |
|---|---|---|---|
| baseline_seeds | 15 | 5–10 | ~2 h |
| backbones | 24 | 10–40 (ViT at 224 px is the expensive one) | ~8 h |
| augmentation_ablation | 30 | 5–10 | ~4 h |
| lr_grid | 18 | 5–10 | ~2.5 h |

Submitted as SLURM job arrays with `%8` concurrency; see `cluster/README.md`.
On a laptop (M-series, MPS) each small-CNN run is ~15–25 min, so Q1 and Q4
are feasible locally; Q2 is not.

## Known gaps to close before writing anything up

* Ensembles (`src/models/ensemble.py`) are implemented but not yet driven by
  a sweep; an "ensemble of the 5 seeds" row for each model is a cheap and
  informative addition.
* Grad-CAM (`src/evaluation/explainability.py`) is not yet part of
  `evaluate.py --analysis`.
* The traditional-ML results (`results/Traditional_ML_Algo_results`) are
  single-run numbers from the notebooks and should be re-run with seeds for
  a fair comparison table.
