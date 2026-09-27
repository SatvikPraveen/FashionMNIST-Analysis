# Usage guide

How to do each common task. Every flag of every tool is listed in the
[command-line reference](CLI_REFERENCE.md), which is generated from the tools
themselves. What each module offers is in the [feature guide](FEATURES.md).

## Contents

1. [Set up](#set-up)
2. [Prepare the data](#prepare-the-data)
3. [Train](#train)
4. [Evaluate one model](#evaluate-one-model)
5. [Run a study with several seeds](#run-a-study-with-several-seeds)
6. [Serve a model](#serve-a-model)
7. [Configuration](#configuration)
8. [Where outputs go](#where-outputs-go)
9. [Troubleshooting](#troubleshooting)

## Set up

```bash
git clone https://github.com/SatvikPraveen/FashionMNIST-Analysis.git
cd FashionMNIST-Analysis
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -q
```

`requirements.txt` includes the notebooks, apps and serving stack. For
training only (for example on a cluster), install PyTorch for your CUDA version
and then `pip install -r cluster/requirements-cluster.txt`.

The device is chosen automatically: CUDA, then Apple MPS, then CPU. Add
`--force-cpu` to `train.py` to override it.

## Prepare the data

```bash
python src/cli/prepare_data.py
```

This downloads Fashion-MNIST to `data/` and writes seeded
`fashion_mnist_{train,val,test}.csv` files to `data/processed/`: 48,000
training and 12,000 validation images from the official training set, and the
official 10,000-image test set, plus `split.json` recording how the split was
made. Training does not need the CSVs: it reads the torchvision copy. Both use
the same split function, so for the same seed (`--seed`, default 42) the
validation CSV contains exactly the images training held out.

## Train

```bash
# the three custom CNNs; the best one is copied to models/best_model_weights/
python src/cli/train.py --model all --seed 0

# one model, with mixed precision on a GPU
python src/cli/train.py --model resnet --seed 0 --amp

# an ImageNet-pretrained backbone from timm (any timm id works, or an alias)
python src/cli/train.py --model vit_tiny --pretrained --epochs 30 --lr 3e-4 --set training.optimizer=adamw

# train only the classifier head of a pretrained backbone
python src/cli/train.py --model convnext_tiny --pretrained --freeze-backbone --epochs 10
```

**Change any setting for one run** with `--set KEY=VALUE`, where the key is a
path in `config.yaml` and the value is parsed as YAML. For example, the best
recipe found for the small CNNs drops random cropping:

```bash
python src/cli/train.py --model tinyvgg --set augmentation.random_crop=false
python src/cli/train.py --model tinyvgg --set training.label_smoothing=0.1 --set augmentation.mixup=false
```

**Reproducibility.** `--seed` fixes model initialisation, data order, the
train/validation split and augmentation. `--deterministic` additionally forces
deterministic GPU kernels, which is slower and only needed for bit-exact
reruns.

**Resume.** Every epoch writes `<model>_last.pt`. Re-running the same command
with `--resume` continues from the last completed epoch, and a larger
`--epochs` extends the run with the learning-rate schedule rebuilt for the new
budget.

**Tracking.** Every run writes `run.json` and `metrics.jsonl` (see
[Where outputs go](#where-outputs-go)). Add `--mlflow` or `--wandb` to mirror
them to those services if installed.

`src/cli/finetune.py` runs a small sequential grid over learning rate, batch
size and patience on one machine. For anything larger, use a sweep instead.

## Evaluate one model

```bash
python src/cli/evaluate.py \
  --model_path models/best_model_weights/best_model_weights.pth \
  --test_csv   data/processed/fashion_mnist_test.csv
```

This writes accuracy, precision, recall and F1, predictions, a confusion
matrix and a grid of example predictions. The architecture is read from the
checkpoint's `_spec.json`, or from `best_model_info.json` for the shipped best
model, so timm backbones evaluate the same way as the custom CNNs.

Add the full analysis with:

```bash
python src/cli/evaluate.py \
  --model_path runs/baseline_fixed/tinyvgg_seed0/tinyvgg/tinyvgg_best.pth \
  --test_csv   data/processed/fashion_mnist_test.csv \
  --val_csv    data/processed/fashion_mnist_val.csv --analysis
```

`--analysis` adds calibration (ECE, NLL, Brier, and temperature scaling
fitted on `--val_csv`), per-class metrics, and accuracy under seven
corruptions at five severities, with figures and an `analysis.json`.
`--no_robustness` skips the corruption sweep, which is the slow part on a CPU.

The temperature must be fitted on images the model did not train on. The
evaluator reads the model's training seed from its `run.json` and compares it
with the seed in `split.json`: when they match it uses `--val_csv`, otherwise
it rebuilds the model's own validation split from its seed. If the seed is
unknown it warns that the CSV may overlap the training data.

## Run a study with several seeds

A study is a YAML file in `sweeps/`. It lists models, seeds, named variants
(sets of config overrides) and an optional grid, and expands to one run per
combination:

```yaml
# sweeps/my_study.yaml
name: my_study
output_root: runs/my_study
models: [tinyvgg, resnet]
seeds: [0, 1, 2, 3, 4]
base_args: ["--amp"]                 # passed to every train.py call
variants:
  default: {}
  no_crop:
    augmentation.random_crop: false
grid:                                # optional cartesian product
  training.learning_rate: [1.0e-3, 3.0e-4]
```

```bash
python src/cli/sweep.py expand sweeps/my_study.yaml      # writes the manifest, prints the run count
python src/cli/sweep.py run    sweeps/my_study.yaml --all  # run everything here, one after another
python src/cli/sweep.py status sweeps/my_study.yaml      # finished / failed / pending
```

On a SLURM cluster, each manifest row is one job-array task; see
[`cluster/README.md`](../cluster/README.md). Runs resume automatically, so
failed or pre-empted tasks can simply be resubmitted.

Summarise and analyse the results:

```bash
# mean, std and 95% CI per group
python src/cli/aggregate.py runs/my_study --out results/sweeps/my_study

# compare every group with one baseline, paired by seed
python src/cli/aggregate.py runs/my_study --baseline 'tinyvgg|default'

# calibration and robustness for every checkpoint, and seed ensembles per group
python src/cli/analyze_runs.py  runs/my_study --out results/sweeps/my_study
python src/cli/ensemble_runs.py runs/my_study --out results/sweeps/my_study
```

Group names are `model|variant`, with the grid values appended when there is a
grid. The paired comparison only uses seeds both groups share, which is why
studies reuse the same seeds across variants.

After adding results to `results/sweeps/`, rebuild the website locally with
`python site/build.py` (it writes `_site/index.html`); pushing to `main`
rebuilds and deploys it automatically.

## Serve a model

```bash
uvicorn src.serving.api:app --port 8000          # REST API, docs at http://localhost:8000/docs
python apps/gradio_app.py                        # Gradio demo
streamlit run apps/streamlit_dashboard.py        # Streamlit dashboard
docker compose -f docker/docker-compose.yml up   # all of the above plus Jupyter
```

Initialise the API with a checkpoint written by `train.py`:

```bash
curl -X POST "http://localhost:8000/initialize?model_path=models/best_model_weights/best_model_weights.pth&config_path=config.yaml"
curl -F "file=@shirt.png" http://localhost:8000/predict
```

Input images can be any size, colour or grayscale. They are converted to the
28 × 28 grayscale format the models were trained on, and photos with a light
background are inverted to match Fashion-MNIST's light-on-black images. The
apps offer the shipped best model plus any model you have trained, and say so
when a model has no trained weights.

## Configuration

All defaults live in `config.yaml`. The settings most worth knowing:

| Key | Default | Meaning |
|---|---|---|
| `training.epochs` | 75 | upper limit; early stopping often ends runs well before it |
| `training.early_stopping_patience` | 10 | epochs without validation-accuracy improvement before stopping |
| `training.learning_rate`, `weight_decay` | 1e-3, 1e-4 | the optimum in the learning-rate study |
| `training.optimizer`, `scheduler` | adam, cosine | also `adamw`, `sgd`; `step`, `exponential`, `none` |
| `training.seed`, `deterministic` | 42, false | see Reproducibility above |
| `training.amp`, `compile` | false, false | mixed precision and `torch.compile` (CUDA) |
| `training.label_smoothing`, `grad_clip` | 0.0, null | optional regularisation |
| `model.pretrained`, `image_size` | true, null | timm backbones only; null means 64 px for CNNs, 224 px for ViT |
| `augmentation.random_crop` | true | padded crop; helps ResNet, hurts the two small CNNs |
| `augmentation.rotation`, `horizontal_flip` | 10, true | per-image rotation (degrees) and flip |
| `augmentation.mixup`, `cutmix`, `mix_prob` | true, true, 0.3 | batch mixing and how often it is applied |
| `augmentation.fill` | background | fill for pixels exposed by crop and rotation |
| `augmentation.legacy_batch_mode` | false | reproduce the pre-fix pipeline; only for comparisons |
| `data.num_workers` | 0 | data-loader workers; raise on machines with many cores |
| `monitoring.mlflow_tracking`, `wandb_enabled` | false | mirror run metrics to those services |

## Where outputs go

```text
models/all_models/<model>/          # train.py default output directory
    <model>_best.pth                # best-validation weights
    <model>_best_spec.json          # architecture, so any tool can rebuild the model
    <model>_last.pt                 # full state for --resume
    <model>_history.json            # per-epoch curves
    run.json, metrics.jsonl         # run metadata and metrics
models/best_model_weights/          # best of a `--model all` run, plus best_model_info.json
runs/<study>/<run>/<model>/         # the same files for every sweep run
results/sweeps/                     # aggregated tables and per-run CSVs (committed)
results/evaluation_results/         # evaluate.py CSVs and analysis.json
figures/evaluation_plots/           # evaluate.py figures
```

## Troubleshooting

**Out of GPU memory.** Lower `--batch-size`, or use `--amp`. ViT models at
224 px are by far the most memory-hungry.

**Training is slow on a CPU.** Use `minicnn` or `tinyvgg`, fewer `--epochs`,
and `--num-workers` above 0.

**MPS is not used on an Apple Silicon Mac.** Check that
`python -c "import torch; print(torch.backends.mps.is_available())"` prints
`True`; upgrade PyTorch if not.

**A pretrained backbone fails to download.** On machines without internet
access, run `python cluster/prefetch.py` on a connected machine first and share
`HF_HOME` / `TORCH_HOME`; see [`cluster/README.md`](../cluster/README.md).

**"could not create a primitive" on a CPU-only machine.** Some virtual machines
lack CPU features that PyTorch's oneDNN backend needs. The code detects this
and falls back to PyTorch's own kernels automatically.

**A test fails after changing a command-line flag.** Run
`python docs/cli_reference.py` to regenerate the command-line reference.
