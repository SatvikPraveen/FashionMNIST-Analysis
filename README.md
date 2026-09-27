# Fashion-MNIST Analysis

![Tests](https://github.com/SatvikPraveen/FashionMNIST-Analysis/actions/workflows/tests.yml/badge.svg)
![MIT License](https://img.shields.io/github/license/SatvikPraveen/FashionMNIST-Analysis)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![Repo Size](https://img.shields.io/github/repo-size/SatvikPraveen/FashionMNIST-Analysis)
![Issues](https://img.shields.io/github/issues/SatvikPraveen/FashionMNIST-Analysis)
![Stars](https://img.shields.io/github/stars/SatvikPraveen/FashionMNIST-Analysis?style=social)

> An in-depth exploration of fashion item classification using the Fashion MNIST dataset.

This project covers data exploration, classical baselines, CNN training and fine-tuning for classifying Fashion-MNIST images.

---

## Overview

This project focuses on analyzing the **Fashion MNIST** dataset using various Convolutional Neural Networks (CNNs), including **MiniCNN**, **TinyVGG**, and **ResNet**. The models were trained with a production-ready pipeline featuring correct Fashion-MNIST normalization, val_acc-based early stopping, and automatic best-model selection. **TinyVGG** achieved the highest performance across all three architectures.

---

## Key Features

- **Traditional Machine Learning Models**: Classification using **Random Forest**, **k-Nearest Neighbors**, and **XGBoost**, combined with dimensionality reduction techniques like PCA, t-SNE, and UMAP.
- **Custom Baseline Models**: Implementation of lightweight **MiniCNN**, **TinyVGG**, and **ResNet** architectures.
- **Fine-Tuning Pipeline**: A pipeline to tune hyperparameters such as learning rates, batch sizes, and early stopping patience values.
- **Visualization Tools**:
  - Confusion matrices for baseline and fine-tuned models.
  - Sample predictions for visual validation.
- **Metrics Comparison**: Detailed comparison of baseline vs. fine-tuned models using accuracy, precision, recall, and F1-score.
- **Reproducibility**: Scripts are modular and adaptable for other datasets.

---

## Project Structure

```bash
FashionMNIST-Analysis/
├── data/
│   ├── processed/          # Train/val/test CSVs (fashion_mnist_{train,val,test}.csv)
│   └── FashionMNIST/raw/   # Raw binary files downloaded by torchvision
├── eda/
│   └── EDA.ipynb           # Exploratory data analysis notebook
├── figures/
│   ├── EDA_plots/          # EDA visualizations
│   ├── evaluation_plots/   # Confusion matrix and prediction visualizations
│   ├── modeling_plots/     # Plots generated during model training
│   └── Traditional_ML_Algo_plots/  # Traditional ML confusion matrices
├── models/
│   ├── all_models/
│   │   ├── minicnn/        # MiniCNN checkpoint + training history
│   │   ├── tinyvgg/        # TinyVGG checkpoint + training history
│   │   └── resnet/         # ResNet checkpoint + training history
│   └── best_model_weights/ # Best overall model weights + best_model_info.json
├── notebooks/              # Jupyter notebooks for the workflow
├── results/
│   ├── evaluation_results/ # predictions_vector.csv, evaluation_metrics.csv
│   └── Traditional_ML_Algo_results/
├── src/
│   ├── cli/                # train.py, evaluate.py, finetune.py, prepare_data.py
│   ├── data/               # dataset.py, augmentation.py, preparation.py
│   ├── models/             # architectures.py, ensemble.py, transfer.py
│   ├── training/           # trainer.py, tuner.py, utils.py
│   ├── evaluation/         # evaluate.py, metrics.py, explainability.py
│   └── serving/            # inference.py, api.py
├── tests/                  # Unit tests (test_models.py, test_inference.py, test_utils.py)
├── apps/                   # Streamlit dashboard and Gradio app
├── docker/                 # Dockerfile and docker-compose.yml
├── docs/                   # FEATURES.md, DEPLOYMENT.md, USAGE_GUIDE.md, etc.
├── config.yaml             # All training/augmentation hyperparameters
├── requirements.txt
└── setup_project.py
```

---

### Description of Key Components

- **`data/`**: Directory for storing raw data.
- **`data_preparation/`**: Processed CSV files for training, validation, and testing datasets.
- **`eda/`**: Designated folder for EDA
  - **`EDA.ipynb`**: For exploratory data analysis.
- **`figures/`**:
  - **`EDA_plots/`**: Visualizations from Exploratory Data Analysis.
  - **`evaluation_plots/`**: Visualizations and metrics for model evaluation.
  - **`modeling_plots/`**: Figures generated during model training.
  - **`fine_tuning_plots/`**: Figures generated during fine-tuning of models.
- **`models/`**:
  - **`all_models/`**: Contains saved weights for all trained models.
  - **`best_model/`**: Contains the final best model after evaluation.
  - **`best_model_weights/`**: Weights of the best-performing model.
- **`notebooks/`**: Jupyter notebooks for the entire workflow:
  - **`modeling.ipynb`**: For training baseline models.
  - **`finetuning.ipynb`**: For fine-tuning models with hyperparameter optimization.
  - **`evaluation.ipynb`**: For evaluation and comparison of models.
- **`src/`**:
  - **`model_definitions.py`**: Contains all model architecture definitions.
  - **`utils.py`**: Utility functions for training, testing, and evaluation.
  - **`evaluation.py`**: Handles model evaluation, including metrics calculation and prediction visualization.
- **`README.md`**: Project documentation and execution details.
- **`requirements.txt`**: Required libraries and dependencies for the project.
- **`setup_project.py`**: Script for creating the project directory structure.
- **`main.py`**: Evaluates the best-trained model, generating predictions, metrics, and visualizations.

---

## Dataset

- **Fashion MNIST** is a dataset of Zalando’s article images, consisting of **60,000 training** and **10,000 testing** grayscale images in **10 classes**.
- Each image is **28x28 pixels**.

![EDA Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/EDA_plots/sample_images_grid.png)

## Class Labels

Below are the 10 class labels for the Fashion MNIST dataset:

| Label | Class |
| ----- | ----------- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## Steps in the Workflow

### 1. Exploratory Data Analysis

- Distribution of labels.
- Sample image visualizations.
- Data normalization and preprocessing.

### 2. Traditional Machine Learning

- Models Used:
  - **Random Forest**
  - **k-Nearest Neighbors**
  - **XGBoost**
- Dimensionality Reduction Techniques:
  - **PCA**: Principal Component Analysis for feature reduction.
  - **t-SNE**: Non-linear dimensionality reduction for visualization.
  - **UMAP**: Uniform Manifold Approximation and Projection for clustering and analysis.
- Hyperparameter Tuning:
  - Grid search optimization for all models.
- **Evaluation Metrics**:
  - Confusion matrices, accuracy, and classification reports for each model.

### 3. CNN Baseline Modeling

- Architectures implemented:
  - **MiniCNN**: A lightweight custom CNN.
  - **TinyVGG**: Inspired by VGG architecture, with fewer layers.
  - **ResNet**: Residual Network with skip connections for better gradient flow.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score for all models.

### 4. Fine-Tuning

- **Hyperparameter Grid**:
  - Learning Rates: `[1e-5, 5e-6]`
  - Batch Sizes: `[32, 64]`
  - Early Stopping Patience: `[2, 3]`
- **Best Model Selection**:
  - **TinyVGG** achieved the highest test accuracy (**93.21%**) and was automatically saved to `models/best_model_weights/`.

### 5. Evaluation

- Confusion matrix and prediction visualization for the best model (TinyVGG).
- **Evaluation Metrics**:
  - MiniCNN test accuracy: **89.93%**
  - ResNet test accuracy: **91.46%**
  - **TinyVGG test accuracy: 93.21%** (best)
- **Visualization**:
  - Sample predictions from the best TinyVGG model.

---

## Key Visualizations

### Confusion Matrix - Best Model (TinyVGG, 93.21% test accuracy)

![Confusion Matrix](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/confusion_matrix.png)

### Prediction Visualization

Sample predictions from the best TinyVGG model:

![Prediction Visualization](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/figures/evaluation_plots/prediction_visualization.png)

---

## Results

| Metric        | MiniCNN | ResNet | **TinyVGG (Best)** |
| ------------- | ------- | ------ | ------------------ |
| **Accuracy**  | 0.8993  | 0.9146 | **0.9321**         |
| **Precision** | 0.8995  | 0.9149 | **0.9323**         |
| **Recall**    | 0.8993  | 0.9146 | **0.9321**         |
| **F1-Score**  | 0.8992  | 0.9146 | **0.9321**         |

> **Note (2026-09):** these are single-seed numbers from the original
> pipeline, whose accuracy was a mean of per-batch accuracies (the recorded
> 0.9321 is exactly 9336/10016, i.e. the 16-image final batch was weighted
> like a full batch of 32). The trainer now reports true sample-level
> accuracy. The multi-seed table below supersedes this one.

> **⚠ Augmentation bug found on 2026-09-27 (fixed in `a341854`).** Every
> *augmented* run in the tables below used an augmentation pipeline that
> (1) applied one random crop / flip / rotation to the **whole batch** instead
> of per image, and (2) padded crops and rotations with **mid-grey** instead of
> the black background. The tables marked "legacy pipeline" are reported
> as measured. Both the ablation and the baseline were **re-run on the fixed
> pipeline with the old one as a paired control** (sections below). The fix
> matters most for the largest CNN: **ResNet gains +1.55 points and its seed
> spread shrinks 7×, so ResNet now clearly beats TinyVGG** (the legacy "they
> tie" conclusion was an artefact). For TinyVGG the fix is worth only +0.36
> to +0.48 points, and the augmentation findings (crop hurts, the
> ensemble-calibration interaction) hold on the fixed pipeline.

### Multi-seed baseline (5 seeds each, 2026-09-27, legacy augmentation pipeline)

Official 10,000-image test set, best-validation checkpoint evaluated once
per run, bf16 mixed precision, full augmentation recipe from `config.yaml`.
Produced by `sweeps/baseline_seeds.yaml` → `src/cli/aggregate.py`
([raw per-run CSV](results/sweeps/baseline_seeds_runs.csv)).

| Model | n | Test acc (mean ± std) | 95% CI | min | max | Params | Time / run |
|---|---|---|---|---|---|---|---|
| ResNet-18 (custom) | 5 | 0.9256 ± 0.0134 | ±0.0166 | 0.9061 | 0.9400 | 11.17M | 12.1 min |
| TinyVGG | 5 | 0.9225 ± 0.0057 | ±0.0071 | 0.9134 | 0.9274 | 0.14M | 4.5 min |
| MiniCNN | 5 | 0.9000 ± 0.0034 | ±0.0043 | 0.8975 | 0.9060 | 0.11M | 3.9 min |

What the seeds change about the story (legacy pipeline; **superseded by the
fixed-pipeline baseline below**): the single-run claim that TinyVGG
beats ResNet is **not supported**. The two confidence intervals overlap,
and ResNet's seed-to-seed spread (0.906–0.940) is more than twice
TinyVGG's. MiniCNN is clearly behind both. TinyVGG remains the best
accuracy per parameter by a wide margin (80× fewer parameters than ResNet
for a statistically indistinguishable result).

**Calibration and robustness of the same 15 checkpoints**
(`src/cli/analyze_runs.py`; temperature fitted on each run's own validation
split; robustness = error under 7 corruptions × 5 severities;
[summary](results/sweeps/baseline_seeds_analysis_summary.md),
[per run](results/sweeps/baseline_seeds_analysis_runs.csv)):

| Model | ECE | ECE after T | T | Mean corruption error | Contrast err | Brightness err | Noise err |
|---|---|---|---|---|---|---|---|
| ResNet-18 (custom) | 0.015 ± 0.012 | 0.009 | 0.86 | **0.272 ± 0.010** | **0.265** | **0.293** | 0.529 |
| TinyVGG | 0.012 ± 0.002 | 0.009 | 0.86 | 0.310 ± 0.022 | 0.437 | 0.456 | 0.509 |
| MiniCNN | 0.015 ± 0.002 | 0.010 | 0.84 | 0.338 ± 0.017 | 0.510 | 0.533 | **0.403** |

All three are already well calibrated, and slightly *under*confident
(T < 1), the usual signature of Mixup/CutMix training. The robustness
column changes the ResNet-vs-TinyVGG picture: they tie on clean accuracy,
but ResNet is clearly more robust, almost entirely because it tolerates
contrast and brightness shifts far better. ResNet is the only one of the
three with batch normalisation, which is a plausible (untested) cause.
MiniCNN is, oddly, the most robust to Gaussian noise. The top confusion for
every model is Shirt ↔ T-shirt/top.

### Multi-seed baseline on the fixed pipeline (5 seeds each, 2026-09-27)

Same seeds 0–4 and recipe as the legacy baseline above, with per-image
augmentation and black-background fill
([summary](results/sweeps/baseline_fixed_summary.md),
[runs](results/sweeps/baseline_fixed_runs.csv)).

| Model | Legacy pipeline | **Fixed pipeline** | 95% CI | Δ (paired by seed) | p | Seeds better | Params |
|---|---|---|---|---|---|---|---|
| ResNet-18 (custom) | 0.9256 ± 0.0134 | **0.9412 ± 0.0018** | ±0.0023 | +0.0155 | 0.071 | 4 / 5 | 11.17M |
| TinyVGG | 0.9225 ± 0.0057 | 0.9273 ± 0.0035 | ±0.0044 | +0.0048 | 0.123 | 4 / 5 | 0.14M |
| MiniCNN | 0.9000 ± 0.0034 | 0.9090 ± 0.0042 | ±0.0052 | +0.0089 | 0.016 | 5 / 5 | 0.11M |

**Findings.** The augmentation bug cost every model accuracy and cost the
largest one the most: ResNet gains 1.55 points and its seed-to-seed spread
falls from 0.0134 to 0.0018. On the fixed pipeline **ResNet clearly beats
TinyVGG** (non-overlapping 95% CIs), reversing the legacy conclusion that the
two tie; that "tie" was produced by the bug's extra variance. TinyVGG keeps a
large accuracy-per-parameter advantage (80× fewer parameters for 1.4 points
less).

### Augmentation ablation (TinyVGG, 5 seeds per variant, 2026-09-27, legacy pipeline)

One component removed at a time from the `config.yaml` recipe. Because every
variant uses the same seeds 0–4, differences are tested **paired by seed**
(`aggregate.py --baseline 'tinyvgg|full'`), which removes seed-to-seed noise.
Raw data: [runs](results/sweeps/augmentation_ablation_runs.csv),
[paired table](results/sweeps/augmentation_ablation_paired.md).

| Variant | Test acc (mean ± std) | Δ vs full (paired) | 95% CI of Δ | p | Seeds better |
|---|---|---|---|---|---|
| no random crop | 0.9333 ± 0.0047 | **+0.0095** | [+0.0027, +0.0162] | **0.018** | 5 / 5 |
| no rotation | 0.9293 ± 0.0052 | +0.0055 | [−0.0047, +0.0157] | 0.211 | 4 / 5 |
| no horizontal flip | 0.9258 ± 0.0047 | +0.0020 | [−0.0088, +0.0129] | 0.629 | 3 / 5 |
| full recipe | 0.9238 ± 0.0072 | — | — | — | — |
| no Mixup/CutMix | 0.9229 ± 0.0062 | −0.0009 | [−0.0144, +0.0126] | 0.863 | 3 / 5 |
| no augmentation | 0.9173 ± 0.0015 | −0.0065 | [−0.0160, +0.0029] | 0.127 | 1 / 5 |

**Finding.** Padded random cropping (pad 4, crop 28) *hurts* TinyVGG: removing
it improved accuracy on every seed. No other single component has a detectable
effect at n = 5, and the full recipe beats no augmentation by only 0.65 points
(not significant). **Caveat:** the full recipe trained longest (60 epochs on
average vs a 75-epoch cap), so part of the crop penalty may be an
under-training effect of the fixed budget.

**Robustness of the ablation checkpoints**
([summary](results/sweeps/augmentation_ablation_analysis_summary.md)) points
at the cause. With crop the model is robust to translation (error 0.176 vs
0.432 without) but fragile to contrast (0.439 vs 0.237), brightness (0.449 vs
0.260) and noise (0.520 vs 0.367); mean corruption error is 0.312 with crop
and 0.263 without. That pattern led to the augmentation code, where the
crop's padding turned out to be mid-grey rather than black, and applied
per batch rather than per image (see the warning above). I first suspected
the crop penalty was a symptom of that bug; the fixed-pipeline re-run below
shows it is **not**: cropping hurts even when implemented correctly.

**Replication under the legacy pipeline** (`sweeps/crop_confirmation.yaml`:
fresh seeds 5–9, 150-epoch budget, patience 15;
[summary](results/sweeps/crop_confirmation_summary.md)). The penalty
replicates and is larger on the smaller model: removing crop gains
**+2.42 points for MiniCNN** (95% CI [+1.90, +2.94], p < 0.001, 5/5 seeds)
and **+1.32 points for TinyVGG** ([+0.67, +1.96], p = 0.005, 5/5). Every run
early-stopped between 30 and 90 epochs, far below the 150 cap, so the
under-training explanation is ruled out. (The ResNet no-crop runs were
cancelled once the bug was found.) Under the legacy pipeline the crop
penalty is therefore real and robust; the fixed-pipeline re-run below shows
the bug was not its cause.

### Augmentation ablation on the fixed pipeline (TinyVGG, 5 seeds, 2026-09-27)

Per-image random draws and black-background fill (commit `a341854`), same
seeds 0–4 as the legacy ablation, plus a `legacy` variant that reproduces the
old pipeline exactly (its seed-0 run matched the original run to four decimal
places). Paired by seed against the fixed full recipe
([summary](results/sweeps/augmentation_fixed_summary.md),
[paired](results/sweeps/augmentation_fixed_paired.md)).

| Variant | Test acc (mean ± std) | Δ vs fixed full | 95% CI of Δ | p | Seeds better |
|---|---|---|---|---|---|
| no random crop | **0.9362 ± 0.0036** | +0.0100 | [−0.0004, +0.0204] | 0.055 | 4 / 5 |
| no rotation | 0.9324 ± 0.0031 | +0.0062 | [−0.0012, +0.0135] | 0.080 | 5 / 5 |
| **full recipe (fixed)** | 0.9262 ± 0.0060 | — | — | — | — |
| no Mixup/CutMix | 0.9257 ± 0.0060 | −0.0006 | [−0.0119, +0.0108] | 0.898 | 2 / 5 |
| no horizontal flip | 0.9233 ± 0.0063 | −0.0029 | [−0.0148, +0.0089] | 0.528 | 2 / 5 |
| legacy (buggy) pipeline | 0.9227 ± 0.0071 | −0.0036 | [−0.0105, +0.0034] | 0.228 | 1 / 5 |
| no augmentation | 0.9169 ± 0.0031 | −0.0094 | [−0.0204, +0.0016] | 0.077 | 1 / 5 |

**Findings.** (1) Fixing the bug is worth only +0.36 points (not
significant): real and worth fixing, but not what drove any earlier
conclusion. (2) **Geometric augmentation hurts this model even when
correct**: removing the crop gains +1.0 points and removing rotation +0.6
points (better on all 5 seeds). A plausible reason is that Fashion-MNIST items
are already centred and size-normalised, so 4-pixel shifts and ±10° rotations
mostly move the training distribution away from the test distribution.
(3) Flip and Mixup/CutMix have no detectable effect; augmentation as a whole
is worth about +0.9 points. The best recipe measured is the fixed pipeline
**without random crop**, 0.9362 ± 0.0036. Its robustness picture is unchanged from
legacy: dropping the crop also gives the lowest mean corruption error
(0.260 vs 0.296 for the fixed full recipe;
[analysis](results/sweeps/augmentation_fixed_analysis_summary.md)).

### Pretrained timm backbones vs training from scratch (3 seeds each, 2026-09-27, legacy pipeline)

ImageNet weights via timm, grayscale input via `in_chans=1`, 28 px images
upsampled to 64 px (CNNs) or 224 px (ViT), AdamW, 30 epochs, bf16. Paired by
seed against the same backbone trained from scratch
([summary](results/sweeps/backbones_summary.md), [runs](results/sweeps/backbones_runs.csv)).

| Backbone | Params | Pretrained | From scratch | Δ from pretraining (paired) | p | Time / run |
|---|---|---|---|---|---|---|
| ViT-Tiny/16 (224 px) | 5.4M | **0.9525 ± 0.0025** | 0.7965 ± 0.0461 | +0.1560 | 0.028 | 15 min |
| ConvNeXt-Tiny | 27.8M | 0.9506 ± 0.0012 | 0.9189 ± 0.0014 | +0.0317 | 0.002 | 19 min |
| EfficientNet-B0 | 4.0M | 0.9397 ± 0.0021 | 0.9020 ± 0.0084 | +0.0377 | 0.011 | 18 min |
| ResNet-18 | 11.2M | 0.9341 ± 0.0019 | 0.9208 ± 0.0026 | +0.0133 | 0.019 | 7 min |

**Findings.** ImageNet pretraining helps every family on every seed, even on
28 px grayscale clothing, and the pretrained ViT-Tiny is the most accurate
model in the study (≈ 2.7 points above the best custom CNN) at only 5.4M
parameters. Trained from scratch the same ViT collapses to 0.80 and is very
unstable across seeds, the familiar result that ViTs lack the inductive bias
to learn well from 48k small images alone. Among from-scratch models none
beats the custom ResNet-18 (0.9256), so on this dataset *pretraining*, not
architecture, is what moves accuracy past ~93%.

### Seed ensembles (5 members per group, 2026-09-27, legacy pipeline)

Softmax average of the five seed checkpoints of each group on the official
test set (`src/cli/ensemble_runs.py`;
[table](results/sweeps/seed_ensembles_ensemble.md)).

| Group | Mean member acc | Best member | Ensemble acc | Gain | ECE: member → ensemble | Mixup/CutMix in training |
|---|---|---|---|---|---|---|
| TinyVGG, no crop | 0.9333 | 0.9398 | **0.9412** | +0.0079 | 0.007 → 0.020 | yes |
| TinyVGG, no rotation | 0.9293 | 0.9360 | 0.9380 | +0.0087 | 0.011 → 0.026 | yes |
| ResNet-18 (custom) | 0.9257 | 0.9400 | 0.9377 | +0.0120 | 0.015 → 0.034 | yes |
| TinyVGG, full recipe | 0.9238 | 0.9282 | 0.9344 | +0.0106 | 0.009 → 0.026 | yes |
| TinyVGG, no Mixup/CutMix | 0.9229 | 0.9294 | 0.9342 | +0.0113 | 0.019 → **0.007** | **no** |
| TinyVGG, no augmentation | 0.9173 | 0.9186 | 0.9339 | **+0.0166** | 0.039 → **0.008** | **no** |
| MiniCNN | 0.9000 | 0.9059 | 0.9110 | +0.0110 | 0.015 → 0.034 | yes |

**Findings.** (1) Ensembling five seeds adds 0.8–1.7 points everywhere and
always lowers NLL. (2) It *worsens* calibration whenever the members were
trained with Mixup/CutMix (ECE roughly triples) and *improves* it sharply when
they were not. Mixup already makes single models underconfident (T < 1 above),
and averaging compounds it. This reproduces Wen et al., *Combining Ensembles
and Data Augmentation Can Harm Your Calibration* (ICLR 2021) on this benchmark.
(3) Augmentation and ensembling are partly substitutes: unaugmented members
disagree most (14.6% of test images) and gain most, so the no-augmentation
ensemble matches the full-recipe ensemble.

**Replication on the fixed augmentation pipeline**
([table](results/sweeps/augmentation_fixed_ensemble.md)). All three findings
hold. Ensembling raises ECE for every Mixup/CutMix-trained group (fixed full
recipe 0.0077 → 0.0208) and lowers it for the groups trained without
Mixup/CutMix (0.0166 → 0.0038) or without any augmentation (0.0331 → 0.0091);
the no-augmentation ensemble (0.9347) again matches the full-recipe ensemble
(0.9352). The most accurate result in the study is a **five-seed ensemble of
TinyVGG trained without random crop: 0.9421**, with 0.14M parameters per member.

### Learning rate × weight decay (TinyVGG, 3 seeds per cell, 2026-09-27, legacy pipeline)

Adam, cosine schedule, full augmentation recipe. Paired by seed against the
`config.yaml` default (LR 1e-3, WD 1e-4). Raw data:
[runs](results/sweeps/lr_grid_runs.csv), [paired table](results/sweeps/lr_grid_paired.md).

| LR | WD | Test acc (mean ± std) | Δ vs default (paired) | p | Seeds better |
|---|---|---|---|---|---|
| 3e-4 | 1e-4 | 0.9268 ± 0.0015 | +0.0006 | 0.424 | 2 / 3 |
| **1e-3** | **1e-4** | **0.9262 ± 0.0021** | — | — | — |
| 3e-4 | 5e-4 | 0.9208 ± 0.0030 | −0.0054 | 0.113 | 0 / 3 |
| 1e-3 | 5e-4 | 0.9185 ± 0.0051 | −0.0077 | 0.190 | 0 / 3 |
| 3e-3 | 5e-4 | 0.8936 ± 0.0106 | −0.0327 | 0.047 | 0 / 3 |
| 3e-3 | 1e-4 | 0.8889 ± 0.0161 | −0.0373 | 0.067 | 0 / 3 |

**Finding.** The default is already at the optimum: LR 1e-3 and 3e-4 are
indistinguishable, LR 3e-3 costs 3–4 points and is also far less stable
across seeds, and the heavier weight decay loses on every seed at every LR
(not significant at n = 3). Tuning LR/WD is therefore not where the
remaining accuracy is.

---

## Updates (2026)

### Training Pipeline

- **End-to-End CLI Scripts**: `train.py`, `finetune.py`, `prepare_data.py`
- **Data Augmentation**: Mixup, CutMix, RandomErasing, torchvision transforms
- **Multi-Device Support**: Auto-detects CUDA, MPS (Apple Silicon M1/M2/M3), or CPU
- **Config-Driven**: All parameters in `config.yaml` for reproducibility
- **Model Checkpointing**: Saves best models automatically with early stopping
- **Logging**: Training history, metrics tracking, JSON outputs

### Quick Start - New Pipeline

```bash
# 1. Prepare data
python src/cli/prepare_data.py

# 2. Train all models (MiniCNN, TinyVGG, ResNet) — best model auto-saved
python src/cli/train.py --model all \
  --use-csv \
  --train-csv data/processed/fashion_mnist_train.csv \
  --val-csv   data/processed/fashion_mnist_val.csv \
  --test-csv  data/processed/fashion_mnist_test.csv

# 3. Evaluate best model (architecture auto-detected from best_model_info.json)
python src/cli/evaluate.py \
  --model_path models/best_model_weights/best_model_weights.pth \
  --test_csv   data/processed/fashion_mnist_test.csv

# 4. Fine-tune best model
python src/cli/finetune.py \
  --model tinyvgg \
  --pretrained models/best_model_weights/best_model_weights.pth
```

**See [USAGE_GUIDE.md](docs/USAGE_GUIDE.md) for complete instructions**

---

## Research Pipeline (2026-09)

The training stack was upgraded from a single-run demo to a reproducible,
cluster-ready experiment pipeline. Everything below is covered by the
`pytest` suite (99 tests).

| Capability | Where |
|---|---|
| Global seeding (Python / NumPy / torch / CUDA / MPS / DataLoader workers / train-val split), `--deterministic` mode | `src/training/reproducibility.py` |
| Sample-weighted accuracy (the old per-batch mean over-weighted the last partial batch) | `src/training/utils.py` |
| One model registry for the custom CNNs **and any timm backbone** (`resnet18`, `resnet50`, `efficientnet_b0`, `convnext_tiny`, `vit_tiny`, `deit_small`, `timm:<id>`), pretrained or scratch, with a spec JSON next to every checkpoint so evaluation can rebuild it | `src/models/registry.py` |
| AMP (bf16/fp16), `torch.compile`, grad clipping, label smoothing, AdamW, pre-emption-safe `--resume` | `src/training/trainer.py` |
| Experiment tracking: `run.json` (git commit, config, host, SLURM ids, device) + `metrics.jsonl` per run; optional MLflow / W&B mirrors | `src/training/experiment.py` |
| Sweeps → SLURM job arrays; aggregation to mean ± std with 95% CI | `src/cli/sweep.py`, `src/cli/aggregate.py`, `sweeps/`, `cluster/` |
| Calibration (ECE, NLL, Brier, temperature scaling), per-class F1 and confusion pairs, robustness to 7 corruptions × 5 severities | `src/evaluation/analysis.py` |

```bash
# any config key can be overridden from the CLI
python src/cli/train.py --model tinyvgg --seed 3 --amp --set training.label_smoothing=0.1

# pretrained backbone at 224 px, head-only fine-tuning
python src/cli/train.py --model vit_tiny --pretrained --freeze-backbone --epochs 10

# multi-seed study: expand -> run (locally or as a SLURM array) -> aggregate
python src/cli/sweep.py expand sweeps/baseline_seeds.yaml
python src/cli/sweep.py run    sweeps/baseline_seeds.yaml --all        # laptop
sbatch --array=0-14%8 cluster/slurm/train_array.sbatch sweeps/baseline_seeds.yaml   # cluster
python src/cli/aggregate.py runs/baseline_seeds --out results/baseline_seeds

# calibration / per-class / robustness report for a checkpoint
python src/cli/evaluate.py --model_path runs/baseline_seeds/tinyvgg_seed0/tinyvgg/tinyvgg_best.pth \
  --test_csv data/processed/fashion_mnist_test.csv --val_csv data/processed/fashion_mnist_val.csv --analysis
```

The study design (questions, sweeps, protocol, compute estimates) is in
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md); the cluster workflow in
[`cluster/README.md`](cluster/README.md).

---

## How to Run

### Environment Setup

Follow these steps to set up the project on your local machine:

1. **Clone this repository**:  
   Clone the FashionMNIST-Analysis repository to your local machine.
   ```bash
   git clone https://github.com/SatvikPraveen/FashionMNIST-Analysis.git
   cd FashionMNIST-Analysis
   ```
2. **Create an environment**:  
   Set up a virtual Python environment to manage dependencies.

   - For Linux/MacOS:

     ```bash
     python -m venv envf
     source envf/bin/activate
     ```

   - For Windows:
     ```bash
     python -m venv envf
     envf\Scripts\activate
     ```

3. **Install dependencies:**

   Install all required Python libraries listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the setup script:**

   Initializes the project by creating necessary directories

   ```bash
   python setup_project.py
   ```

## Execution

- **Exploratory Data Analysis**: `eda.ipynb`
- **Traditional ML Algorithms**:`Traditional_ML_Algo.ipynb`
- **Model Training**: `modeling.ipynb`
- **Fine-Tuning**: `finetuning.ipynb`
- **Evaluation**: `evaluation.ipynb`

---

## Evaluating the Model

### Using the `evaluate.py` CLI

Evaluates the best model. Architecture is auto-detected from `models/best_model_weights/best_model_info.json`.

- **Plots** are saved to `figures/evaluation_plots/`:
  - `confusion_matrix.png`
  - `prediction_visualization.png`
- **CSVs** are saved to `results/evaluation_results/`:
  - `predictions_vector.csv`
  - `evaluation_metrics.csv`

#### Command to Run

```bash
python src/cli/evaluate.py \
  --model_path models/best_model_weights/best_model_weights.pth \
  --test_csv   data/processed/fashion_mnist_test.csv
```

Optional overrides:
- **`--model_name`**: Force architecture (`ResNet`, `TinyVGG`, `MiniCNN`). Auto-detected if omitted.
- **`--figures_dir`**: Override plot output directory (default: `figures/evaluation_plots`).
- **`--results_dir`**: Override CSV output directory (default: `results/evaluation_results`).
- **`--analysis`** (+ optional `--val_csv`): calibration (ECE, NLL, Brier, temperature scaling), per-class metrics and corruption robustness; writes `analysis.json` and four extra figures. `--no_robustness` skips the corruption sweep.

Any checkpoint written by `train.py` carries a `_spec.json`, so timm backbones evaluate the same way as the custom CNNs.

---

## Technologies Used

- **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn, SciPy

  - Essential libraries for data manipulation, statistical analysis, and visualization.

- **Machine Learning and Data Mining**: Scikit-learn

  - Provides tools for traditional ML models and techniques like PCA and t-SNE.

- **Deep Learning (PyTorch)**: PyTorch, TorchVision, Pillow

  - PyTorch and TorchVision are used for designing, training, and evaluating neural networks. Pillow is used for image preprocessing.

- **Jupyter Notebooks for Analysis**: Jupyter, IPython, IPyKernel, Notebook

  - Enables interactive analysis and visualization in Jupyter Notebook environments.

- **Progress Bar**: TQDM

  - Adds progress bars to loops and processes for better tracking.

- **Generating Model Summaries**: TorchInfo

  - Generates detailed summaries of PyTorch models, including layer-wise parameters and memory usage.

- **Dimensionality Reduction and XGBoost**: UMAP-learn, XGBoost
  - UMAP for advanced dimensionality reduction and XGBoost for traditional gradient-boosted models.

---

## Acknowledgments

This project is inspired by the Fashion MNIST dataset provided by Zalando Research. Special thanks to open-source contributors of **PyTorch** and **Scikit-learn** for enabling this work.

For a detailed explanation of this project, refer to the accompanying [blog post](https://medium.com/@meetdheerajreddy/fashion-mnist-analysis-classifying-fashion-with-deep-learning-0ba793ba5234).

---

## Implementation Notebooks

To explore the different stages of the project workflow, you can access the following Jupyter notebooks:

- **[Data Preparation](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/DataPreparation.ipynb)**: Prepares the dataset for performing tasks
- **[Exploratory Data Analysis (EDA)](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/eda/EDA.ipynb)**: Visualizations and preprocessing steps for Fashion MNIST.
- **[Traditional Machine Learning](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/Traditional_ML_Algo.ipynb)**: Implementation of Random Forest, k-NN, and XGBoost models with dimensionality reduction techniques.
- **[Model Training](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/modeling.ipynb)**: Training baseline CNN models like MiniCNN, TinyVGG, and ResNet.
- **[Fine-Tuning](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/finetuning.ipynb)**: Hyperparameter tuning for the CNN models.
- **[Evaluation](https://github.com/SatvikPraveen/FashionMNIST-Analysis/blob/main/notebooks/evaluate_best_model.ipynb)**: Model evaluation, confusion matrices, and metrics comparison.

You can access the full repository [here](https://github.com/SatvikPraveen/FashionMNIST-Analysis).

## Documentation

For guides and documentation, please refer to the `docs/` folder:

- **[RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)** - Research questions, sweeps, evaluation protocol and compute plan.
- **[cluster/README.md](cluster/README.md)** - Running sweeps as SLURM job arrays.
- **[FEATURES.md](docs/FEATURES.md)** - Complete feature documentation (800+ lines) covering all new modules and capabilities.
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Deployment guide for Docker, Kubernetes, and cloud platforms.
- **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Detailed implementation report of the modernization project.
- **[CONTRIBUTING.md](.github/CONTRIBUTING.md)** - Guidelines for contributing to the project.
- **[CODE_OF_CONDUCT.md](.github/CODE_OF_CONDUCT.md)** - Community code of conduct.

---

## Future Work

- Run the four sweeps in `sweeps/` on a GPU cluster and replace the single-seed results table with mean ± CI numbers.
- Drive `src/models/ensemble.py` from a sweep (ensemble of seeds per architecture).
- Add Grad-CAM output to `evaluate.py --analysis`.
- Re-run the traditional-ML baselines with seeds for a fair comparison table.
- Test the best model on unseen real-world data.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

