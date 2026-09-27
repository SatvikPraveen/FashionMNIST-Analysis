# Command-line reference

Generated from each tool's `--help` by `python docs/cli_reference.py`; do not edit by hand.
Task-oriented examples are in the [usage guide](USAGE_GUIDE.md).

## prepare_data.py

Download Fashion-MNIST and write seeded train/val/test CSV splits.

```text
usage: prepare_data.py [-h] [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR]
                       [--train-split TRAIN_SPLIT] [--no-csv] [--seed SEED]

Prepare Fashion MNIST dataset for training

options:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory for raw data (default: ./data)
  --output-dir OUTPUT_DIR
                        Directory for processed CSV files (default: ./data/processed)
  --train-split TRAIN_SPLIT
                        Train/validation split ratio (default: 0.8)
  --no-csv              Skip CSV conversion (only download)
  --seed SEED           Random seed (default: 42)
```

## train.py

Train one or more models; every run writes run.json, metrics.jsonl and checkpoints.

```text
usage: train.py [-h] [--config CONFIG] [--model MODEL [MODEL ...]] [--pretrained]
                [--no-pretrained] [--image-size IMAGE_SIZE] [--freeze-backbone]
                [--output-dir OUTPUT_DIR] [--use-csv] [--train-csv TRAIN_CSV]
                [--val-csv VAL_CSV] [--test-csv TEST_CSV] [--force-cpu] [--seed SEED]
                [--deterministic] [--num-workers NUM_WORKERS] [--epochs EPOCHS]
                [--batch-size BATCH_SIZE] [--lr LR] [--amp] [--compile] [--resume]
                [--run-name RUN_NAME] [--mlflow] [--wandb] [--set KEY=VALUE]
                [--skip-best-selection]

Train FashionMNIST models

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config file
  --model MODEL [MODEL ...]
                        Model(s) to train. Custom CNNs: minicnn, tinyvgg, resnet;
                        'all' = those three. Any timm id or alias also works:
                        resnet18, resnet50, efficientnet_b0, convnext_tiny, vit_tiny,
                        deit_small, timm:<id>. (default: all)
  --pretrained          Use ImageNet weights for timm models (overrides
                        model.pretrained)
  --no-pretrained       Random-init timm models
  --image-size IMAGE_SIZE
                        Input resolution for timm backbones (overrides
                        model.image_size)
  --freeze-backbone     Train only the classifier head of a timm backbone
  --output-dir OUTPUT_DIR
                        Output directory for models
  --use-csv             Use CSV datasets instead of torchvision
  --train-csv TRAIN_CSV
                        Path to training CSV
  --val-csv VAL_CSV     Path to validation CSV
  --test-csv TEST_CSV   Path to test CSV
  --force-cpu           Force CPU usage
  --seed SEED           Random seed (overrides training.seed in config; default 42)
  --deterministic       Force deterministic algorithms (slower; for bit-exact
                        reproduction)
  --num-workers NUM_WORKERS
                        DataLoader worker processes (overrides data.num_workers in
                        config)
  --epochs EPOCHS       Override training.epochs from config
  --batch-size BATCH_SIZE
                        Override training.batch_size from config
  --lr LR               Override training.learning_rate from config
  --amp                 Enable mixed precision (bf16 on supporting GPUs, else
                        fp16+GradScaler)
  --compile             torch.compile the model (CUDA only)
  --resume              Resume from <output-dir>/<model>/<model>_last.pt if present
  --run-name RUN_NAME   Experiment run name (default: <model>_seed<seed>)
  --mlflow              Mirror metrics to MLflow (monitoring.mlflow_tracking)
  --wandb               Mirror metrics to Weights & Biases (monitoring.wandb_enabled)
  --set KEY=VALUE       Override any config key, e.g. --set
                        training.label_smoothing=0.1 --set augmentation.mixup=false
                        (repeatable; values parsed as YAML)
  --skip-best-selection
                        Do not copy the best model to models/best_model_weights/
                        (sweeps)
```

## evaluate.py

Evaluate one checkpoint: metrics, confusion matrix and, with --analysis, calibration and robustness.

```text
[INFO] Using device: mps
usage: evaluate.py [-h] --model_path MODEL_PATH [--model_name MODEL_NAME] --test_csv
                   TEST_CSV [--figures_dir FIGURES_DIR] [--results_dir RESULTS_DIR]
                   [--batch_size BATCH_SIZE] [--analysis] [--val_csv VAL_CSV]
                   [--no_robustness]

Evaluate pre-trained Fashion MNIST models.

options:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        Path to the pre-trained model weights.
  --model_name MODEL_NAME
                        Architecture name: ResNet, TinyVGG, MiniCNN. Auto-detected
                        from best_model_info.json if omitted.
  --test_csv TEST_CSV   Path to the test dataset CSV.
  --figures_dir FIGURES_DIR
                        Directory to save plots (confusion matrix, prediction grid).
  --results_dir RESULTS_DIR
                        Directory to save CSV outputs (predictions, metrics).
  --batch_size BATCH_SIZE
  --analysis            Also run calibration / per-class / robustness analysis (writes
                        analysis.json + figures).
  --val_csv VAL_CSV     Validation CSV for temperature scaling (with --analysis). Used
                        only when its split seed matches the model's; otherwise the
                        model's own validation split is rebuilt from its seed.
  --no_robustness       Skip the corruption sweep in --analysis (7 corruptions x 5
                        severities).
```

## finetune.py

Grid search over learning rate, batch size and patience (sequential, one machine).

```text
usage: finetune.py [-h] [--config CONFIG] --model MODEL [--output-dir OUTPUT_DIR]
                   [--pretrained PRETRAINED]
                   [--learning-rates LEARNING_RATES [LEARNING_RATES ...]]
                   [--batch-sizes BATCH_SIZES [BATCH_SIZES ...]]
                   [--patience-values PATIENCE_VALUES [PATIENCE_VALUES ...]]
                   [--force-cpu] [--seed SEED]

Fine-tune FashionMNIST models

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config file
  --model MODEL         Model to fine-tune: minicnn, tinyvgg, resnet, or any timm
                        id/alias
  --output-dir OUTPUT_DIR
                        Output directory for results
  --pretrained PRETRAINED
                        Path to pretrained model weights
  --learning-rates LEARNING_RATES [LEARNING_RATES ...]
                        Learning rates to try (default: 1e-5 5e-6)
  --batch-sizes BATCH_SIZES [BATCH_SIZES ...]
                        Batch sizes to try (default: 32 64)
  --patience-values PATIENCE_VALUES [PATIENCE_VALUES ...]
                        Early stopping patience values (default: 2 3)
  --force-cpu           Force CPU usage
  --seed SEED           Random seed (overrides training.seed in config; default 42)
```

## sweep.py

Expand a sweep YAML into runs, run them (locally or as a job array) and show progress.

```text
usage: sweep.py [-h] {expand,run,status} ...

Sweep runner: turn a sweep YAML into a manifest of independent training runs
and execute one (or all) of them.

Designed for SLURM job arrays: ``expand`` once on the login node, then each
array task runs ``run --index $SLURM_ARRAY_TASK_ID``. Every run is a normal
``train.py`` invocation, so anything the CLI accepts can be swept.

Sweep file format (see sweeps/*.yaml):

    name: baseline_seeds
    output_root: runs/baseline_seeds        # one sub-dir per run
    config: config.yaml                     # base config (optional)
    base_args: ["--amp", "--use-csv", ...]  # passed to every run (optional)
    models: [minicnn, tinyvgg, resnet]
    seeds: [0, 1, 2, 3, 4]
    variants:                               # named override sets (optional)
      default: {}
      no_mix: {augmentation.mixup: false, augmentation.cutmix: false}
    grid:                                   # cartesian product (optional)
      training.learning_rate: [1e-3, 3e-4]

Usage:
    python src/cli/sweep.py expand sweeps/baseline_seeds.yaml
    python src/cli/sweep.py run    sweeps/baseline_seeds.yaml --index 3
    python src/cli/sweep.py run    sweeps/baseline_seeds.yaml --all      # sequential
    python src/cli/sweep.py status sweeps/baseline_seeds.yaml

positional arguments:
  {expand,run,status}
    expand             write manifest.jsonl/csv under output_root
    run                run one manifest row (or all sequentially)
    status             show finished / failed / pending rows

options:
  -h, --help           show this help message and exit
```

## sweep.py expand

Write the run manifest for a sweep.

```text
usage: sweep.py expand [-h] sweep

positional arguments:
  sweep

options:
  -h, --help  show this help message and exit
```

## sweep.py run

Run one manifest row (--index) or all rows (--all). Arguments after -- go to train.py.

```text
usage: sweep.py run [-h] (--index INDEX | --all) [--only-pending] [--dry-run] sweep

positional arguments:
  sweep

options:
  -h, --help      show this help message and exit
  --index INDEX   row index (e.g. $SLURM_ARRAY_TASK_ID)
  --all           run every row sequentially
  --only-pending  with --all: skip finished rows
  --dry-run       print commands only
```

## sweep.py status

List finished, failed and pending runs.

```text
usage: sweep.py status [-h] sweep

positional arguments:
  sweep

options:
  -h, --help  show this help message and exit
```

## aggregate.py

Summarise runs: mean, std, 95% CI per group; --baseline adds paired-by-seed tests.

```text
usage: aggregate.py [-h] [--metric METRIC] [--group-by {group,model,variant,sweep}]
                    [--out OUT] [--baseline BASELINE]
                    roots [roots ...]

Aggregate the run.json files of a sweep into a results table with mean,
standard deviation and a 95% confidence interval over seeds.

Usage:
    python src/cli/aggregate.py runs/baseline_seeds
    python src/cli/aggregate.py runs/baseline_seeds --metric test_acc --out results/baseline
    python src/cli/aggregate.py runs/a runs/b --group-by model   # pool several sweeps
    python src/cli/aggregate.py runs/augmentation_ablation --baseline 'tinyvgg|full'  # paired by seed

Outputs (with --out PREFIX): PREFIX_runs.csv (one row per run) and
PREFIX_summary.csv + PREFIX_summary.md (one row per group).

positional arguments:
  roots                 sweep output roots (searched recursively for run.json)

options:
  -h, --help            show this help message and exit
  --metric METRIC
  --group-by {group,model,variant,sweep}
  --out OUT             output prefix, e.g. results/baseline
  --baseline BASELINE   group to compare every other group against, paired by seed
                        (e.g. 'tinyvgg|full')
```

## analyze_runs.py

Calibration, per-class and robustness analysis for every checkpoint in a sweep.

```text
usage: analyze_runs.py [-h] [--out OUT] [--no-robustness] [--recompute]
                       [--data-root DATA_ROOT] [--batch-size BATCH_SIZE]
                       [--num-workers NUM_WORKERS]
                       roots [roots ...]

Batch calibration / per-class / robustness analysis over a sweep's checkpoints.

For every finished run under the given roots this rebuilds the model from its
``*_best_spec.json``, recreates *that run's* validation split from its seed
(for temperature scaling) and the official test set, runs
:func:`src.evaluation.analysis.run_full_analysis`, and writes
``<run_dir>/analysis/analysis.json`` plus figures. It then aggregates the
headline metrics by sweep group (mean ± std over seeds).

Usage:
    python src/cli/analyze_runs.py runs/baseline_seeds --out results/sweeps/baseline_seeds
    python src/cli/analyze_runs.py runs/a runs/b --out results/sweeps/ab --no-robustness
    sbatch cluster/slurm/python.sbatch src/cli/analyze_runs.py runs/baseline_seeds --out ...

Outputs with --out PREFIX: PREFIX_analysis_runs.csv (one row per run) and
PREFIX_analysis_summary.{csv,md} (one row per group).

positional arguments:
  roots

options:
  -h, --help            show this help message and exit
  --out OUT             output prefix, e.g. results/sweeps/baseline_seeds
  --no-robustness
  --recompute           ignore existing analysis.json files
  --data-root DATA_ROOT
  --batch-size BATCH_SIZE
  --num-workers NUM_WORKERS
```

## ensemble_runs.py

Seed ensembles per group, compared with their members.

```text
usage: ensemble_runs.py [-h] [--out OUT] [--groups [GROUPS ...]]
                        [--min-members MIN_MEMBERS] [--data-root DATA_ROOT]
                        [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                        roots [roots ...]

Seed ensembles per sweep group: average the softmax of every finished
checkpoint in a group on the official test set, and compare the ensemble
with the average single member.

Deep ensembles of independently seeded networks are the standard, cheap
baseline for both accuracy and calibration gains; every sweep already has
several seeds per group, so this only costs inference.

Usage:
    python src/cli/ensemble_runs.py runs/baseline_seeds --out results/sweeps/baseline_seeds
    python src/cli/ensemble_runs.py runs/backbones --groups 'resnet18|pretrained'

positional arguments:
  roots

options:
  -h, --help            show this help message and exit
  --out OUT             output prefix; writes <out>_ensemble.{csv,md}
  --groups [GROUPS ...]
                        only these groups
  --min-members MIN_MEMBERS
  --data-root DATA_ROOT
  --batch-size BATCH_SIZE
  --num-workers NUM_WORKERS
```
