# Research plan: what does a modern training recipe buy on a small grayscale benchmark?

**Status (2026-09-27):** infrastructure complete and running on a SLURM GPU cluster.
all questions answered, and every affected result re-run on the fixed
augmentation pipeline. See the README for the full write-up. Full result tables live in the README and `results/sweeps/`.

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
| Q2 | Pretraining helps every family (paired, p ≤ 0.028, 3/3 seeds each): ViT-Tiny +15.6, EfficientNet-B0 +3.8, ConvNeXt-Tiny +3.2, ResNet-18 +1.3 points. Best model: pretrained ViT-Tiny, 0.9528 ± 0.0009 on the fixed pipeline (unchanged by the fix). ViT from scratch collapses (0.80). Crop helps ViT slightly (−0.34 without it, p = 0.027). | `results/sweeps/backbones_*` |
| Q3 | Which augmentation components matter? | `sweeps/augmentation_ablation.yaml` | 30 | Δ test acc vs full recipe |
| Q4 | Is the default LR / weight decay near-optimal for the small CNNs? | `sweeps/lr_grid.yaml` | 18 | test acc |
| Q5 | Are the more accurate models also better calibrated and more robust? | `evaluate.py --analysis` on each Q1/Q2 winner | – | ECE, NLL, mean corruption error |

Every sweep varies **seeds** (3–5) so every comparison comes with a
confidence interval. All numbers are on the **official 10,000-image test
set**, evaluated **once** from the best-validation checkpoint.

## Results so far

| # | Answer | Evidence |
|---|---|---|
| Q1 | On the fixed pipeline: ResNet 0.9412 ± 0.0018, TinyVGG 0.9273 ± 0.0035, MiniCNN 0.9090 ± 0.0042 (n = 5); ResNet clearly best (non-overlapping CIs). On the legacy pipeline ResNet and TinyVGG appeared to tie because the augmentation bug cost ResNet 1.55 points and inflated its variance 7×. The original single-seed "TinyVGG is best" claim does not hold on either pipeline. | `results/sweeps/baseline_fixed_*`, `baseline_seeds_*` |
| Q3 | Padded random crop **hurts** TinyVGG: removing it gains +0.95 points (paired, p = 0.018, 5/5 seeds). No other single component is detectable at n = 5; the whole recipe is worth only +0.65 points over none (p = 0.127). | `results/sweeps/augmentation_ablation_*` |
| Q3b | The crop penalty replicates on fresh seeds with a 150-epoch budget (MiniCNN +2.4, TinyVGG +1.3 points, p ≤ 0.005) and **survives the augmentation-bug fix** (+1.0, p = 0.055; rotation also hurts, +0.6). The bug fix itself is worth only +0.36 points. **The effect depends on capacity:** removing crop helps MiniCNN (+1.46, p = 0.001) but hurts ResNet-18 (−0.50, p = 0.029, 0/5 seeds). | `results/sweeps/crop_confirmation_*`, `augmentation_fixed_*` |
| Q4 | The default LR 1e-3 / WD 1e-4 is optimal; LR 3e-4 is equivalent, LR 3e-3 costs 3–4 points, WD 5e-4 loses on every seed. | `results/sweeps/lr_grid_*` |
| Q2 | Pretraining helps every family (paired, p ≤ 0.028, 3/3 seeds each): ViT-Tiny +15.6, EfficientNet-B0 +3.8, ConvNeXt-Tiny +3.2, ResNet-18 +1.3 points. Best model overall: pretrained ViT-Tiny, 0.9525 ± 0.0025. ViT from scratch collapses (0.80). | `results/sweeps/backbones_*` |
| Q5 | Running: `src/cli/analyze_runs.py` over the baseline and ablation checkpoints. | jobs 488426, 488427 |

Statistical note added during the study: because sweeps reuse seeds across
variants, comparisons are now made **paired by seed**
(`aggregate.py --baseline`), which is what made the crop effect detectable
where the unpaired CIs still overlapped.

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

* ~~Ensembles not driven by a sweep~~ — `src/cli/ensemble_runs.py` now builds
  a seed ensemble per group (job 488429 running).
* Grad-CAM (`src/evaluation/explainability.py`) is not yet part of
  `evaluate.py --analysis`.
* The traditional-ML results (`results/Traditional_ML_Algo_results`) are
  single-run numbers from the notebooks and should be re-run with seeds for
  a fair comparison table.
