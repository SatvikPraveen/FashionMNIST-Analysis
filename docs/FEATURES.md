# Feature guide

What the code base provides, with a short working example for each part.
Every Python example on this page is executed by `tests/test_docs_examples.py`,
so the examples stay correct as the code changes.

For command-line flags see the [usage guide](USAGE_GUIDE.md); for how the
modules fit together see the [architecture overview](ARCHITECTURE.md); for
measured results see the [README](../README.md#headline-results).

## Contents

1. [Models](#models)
2. [Training, reproducibility and tracking](#training-reproducibility-and-tracking)
3. [Data augmentation](#data-augmentation)
4. [Evaluation and analysis](#evaluation-and-analysis)
5. [Ensembles](#ensembles)
6. [Experiments at scale](#experiments-at-scale)
7. [Inference and serving](#inference-and-serving)
8. [Monitoring](#monitoring)
9. [Configuration](#configuration)
10. [Testing](#testing)

---

## Models

All models are built through one registry, `src/models/registry.py`. It
accepts the three custom CNNs and any [timm](https://github.com/huggingface/pytorch-image-models)
model, and every model takes Fashion-MNIST tensors of shape `(N, 1, 28, 28)`
and returns logits of shape `(N, 10)`.

| Name | What it is | Parameters |
|---|---|---|
| `minicnn` | two conv blocks, one hidden layer | 0.11M |
| `tinyvgg` | two VGG-style double-conv blocks | 0.14M |
| `resnet` | ResNet-18-style network with batch norm | 11.2M |
| any timm id or alias | e.g. `resnet18`, `resnet50`, `efficientnet_b0`, `convnext_tiny`, `vit_tiny`, `vit_small`, `deit_small`, `timm:<id>` | varies |

timm backbones are built with `in_chans=1`, so pretrained RGB stems are
averaged to grayscale by timm, and are wrapped in `TimmClassifier`, which
upsamples the 28 px input to the backbone's resolution: 64 px for CNNs and
224 px for ViT-style models by default.

```python
import torch
from src.models import build_model, resolve_name

x = torch.randn(4, 1, 28, 28)

cnn = build_model("tinyvgg")
vit = build_model("vit_tiny", pretrained=False)            # pretrained=True downloads ImageNet weights
frozen = build_model("resnet18", freeze_backbone=True)     # only the classifier head trains

print(resolve_name("vit_tiny"))                            # vit_tiny_patch16_224
print(cnn(x).shape, vit(x).shape)                          # torch.Size([4, 10]) twice
print(sum(p.requires_grad for p in frozen.parameters()))   # 2: head weight and bias
```

Every checkpoint written by training has a `<name>_best_spec.json` next to it
describing the architecture, so any model can be rebuilt without knowing
what it was:

```python
import torch, tempfile, os
from src.models import build_model, ModelSpec, load_model_from_checkpoint
from src.models.registry import spec_path_for

d = tempfile.mkdtemp()
weights = os.path.join(d, "tinyvgg_best.pth")
torch.save(build_model("tinyvgg").state_dict(), weights)
ModelSpec(name="tinyvgg").save(spec_path_for(weights))

model = load_model_from_checkpoint(weights)     # reads tinyvgg_best_spec.json
print(type(model).__name__)                     # TinyVGG
```

`src/models/transfer.py` keeps the older `TransferLearningModel` wrapper for
code that used it; new code should use the registry.

## Training, reproducibility and tracking

Training is driven by `src/cli/train.py` (flags in the [usage guide](USAGE_GUIDE.md)).
Its building blocks are importable:

| Feature | Where |
|---|---|
| Seeding of Python, NumPy, torch, CUDA, MPS, data-loader workers and the train/val split; optional deterministic kernels | `src/training/reproducibility.py` |
| Sample-weighted accuracy and loss | `src/training/utils.py` |
| Mixed precision (bf16 where supported, else fp16 with a gradient scaler), `torch.compile`, gradient clipping, label smoothing, Adam/AdamW/SGD, cosine/step/exponential schedules | `src/training/trainer.py` |
| Resumable training: `<model>_last.pt` holds model, optimiser, scaler, history and early-stopping state; the LR schedule is rebuilt for the current epoch budget | `src/training/trainer.py` |
| Per-run `run.json` (git commit, config, host, SLURM ids, device, timings, summary) and `metrics.jsonl` (per epoch); optional MLflow / Weights & Biases mirrors | `src/training/experiment.py` |

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training import set_seed, make_generator
from src.training.utils import test_step
from src.models import build_model

set_seed(0)
x, y = torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,))
loader = DataLoader(TensorDataset(x, y), batch_size=32, generator=make_generator(0))
loss, acc = test_step(build_model("minicnn"), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
print(round(acc * 50))   # accuracy is correct / 50, not an average of the two batch accuracies
```

```python
import tempfile
from src.training.experiment import ExperimentLogger, read_run, read_metrics

d = tempfile.mkdtemp()
with ExperimentLogger(d, run_name="demo", config={"lr": 1e-3}) as log:
    log.log_params({"model": "tinyvgg"})
    log.log_metrics({"val_acc": 0.91}, step=1)
    log.finish({"test_acc": 0.92})
print(read_run(d)["summary"], read_metrics(d)[0]["val_acc"])
```

## Data augmentation

`src/data/augmentation.py`, configured by the `augmentation` section of
`config.yaml`.

- **Geometric:** random rotation, horizontal and vertical flips, padded
  random crop, random affine. Each image gets its own random draw, and pixels
  exposed by padding or rotation are filled with the normalised black
  background. `augmentation.legacy_batch_mode: true` reproduces the pre-fix
  behaviour (one draw per batch, grey fill) for comparisons.
- **Mixing:** Mixup and CutMix, applied to a batch with probability
  `augmentation.mix_prob`.
- **Photometric and occlusion:** colour jitter, Gaussian noise and blur,
  random erasing.

```python
import torch
from src.data.augmentation import AugmentationPipeline, FMNIST_BACKGROUND

pipe = AugmentationPipeline({
    "torchvision": {"rotation": 10, "horizontal_flip": True,
                    "random_crop": {"size": 28, "padding": 4}, "fill": FMNIST_BACKGROUND},
    "mixup": {"alpha": 0.1},
    "cutmix": {"alpha": 1.0},
})
x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
x_aug = pipe.apply_sample_augmentations(x)          # independent draw per image
x_mix, y_a, y_b, lam = pipe.apply_mixup(x_aug, y)   # use Mixup.mixup_criterion for the loss
print(x_aug.shape, 0.0 <= lam <= 1.0)
```

Which of these help is measured in the README: random crop helps the larger
models and hurts the two small CNNs.

## Evaluation and analysis

`src/cli/evaluate.py` gives accuracy, a confusion matrix, per-class metrics
and a prediction grid for one checkpoint; add `--analysis` for the full
report below. The functions in `src/evaluation/analysis.py` work on logits
and labels, so they can be used on any model:

- **Calibration:** expected and maximum calibration error with reliability
  bins, NLL, Brier score, and temperature scaling fitted on a validation set.
- **Per-class:** precision, recall and F1 per class, and the most confused
  class pairs.
- **Robustness:** accuracy under seven corruptions (Gaussian noise, blur,
  contrast, brightness, rotation, translation, occlusion) at five
  severities, and the mean corruption error.

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.analysis import calibration_report, fit_temperature, per_class_report, robustness_sweep
from src.models import build_model

logits, labels = torch.randn(200, 10) * 3, torch.randint(0, 10, (200,))
T = fit_temperature(logits, labels)
report = calibration_report(logits, labels, temperature=T)
print(round(report["ece"], 3), round(report["scaled"]["ece"], 3))

print(per_class_report(logits.argmax(1), labels)["top_confusions"][0]["count"] > 0)

loader = DataLoader(TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))), batch_size=16)
rob = robustness_sweep(build_model("minicnn").eval(), loader, torch.device("cpu"),
                       corruptions=["contrast"], severities=(1, 5))
print(sorted(rob["accuracy"]["contrast"]))          # ['1', '5']
```

Grad-CAM heat maps for convolutional models are in `src/evaluation/explainability.py`:

```python
import torch
from src.evaluation.explainability import GradCAM
from src.models import build_model

model = build_model("tinyvgg").eval()
cam = GradCAM(model, target_layer="conv_block_2").generate_cam(torch.randn(1, 1, 28, 28))
print(cam.shape)   # a 2-D map, overlay with GradCAM.visualize_cam
```

## Ensembles

For trained sweeps, `src/cli/ensemble_runs.py` averages the softmax of every
seed in a group and compares the ensemble with its members, including
calibration. For models in memory, `src/models/ensemble.py` has soft and hard
voting, stacking (a logistic-regression meta-learner) and bagging:

```python
import torch
from src.models import build_model
from src.models.ensemble import EnsembleVoting

members = [build_model("minicnn").eval() for _ in range(3)]
ensemble = EnsembleVoting(members, voting="soft")
print(ensemble.predict(torch.randn(4, 1, 28, 28)).shape)   # (4,)
```

## Experiments at scale

A study is a YAML file in `sweeps/` that expands to independent runs, one per
model × variant × grid point × seed. The same runs execute sequentially on one
machine or as a SLURM job array, and every run can resume.

| Tool | Job |
|---|---|
| `src/cli/sweep.py expand / run / status` | build the run manifest; run one row (`--index`, e.g. a job-array task) or all; show progress |
| `src/cli/aggregate.py` | mean, standard deviation and 95% CI per group; `--baseline GROUP` adds paired-by-seed differences and t-tests |
| `src/cli/analyze_runs.py` | calibration, per-class and robustness analysis for every checkpoint in a sweep |
| `src/cli/ensemble_runs.py` | seed ensembles per group |
| `cluster/` | generic SLURM templates and a script that prefetches the dataset and pretrained weights for offline nodes |

See [`cluster/README.md`](../cluster/README.md) for the workflow.

## Inference and serving

`src/serving/inference.py` turns arbitrary images (files or arrays, any size,
colour or grayscale) into the 28 × 28 grayscale, Fashion-MNIST-normalised input
every model here was trained on, and returns the predicted class, confidence
and top-k. `ImagePreprocessor.for_fashion_mnist()` is the right set-up for any
model from the registry (timm backbones resize internally); with `invert=None`
it inverts photos with a light background to match Fashion-MNIST's
light-on-black images. `load_trained_model` finds a model's saved weights and
rebuilds it:

```python
import numpy as np
from src.serving.inference import ImagePreprocessor, RealWorldInference, load_trained_model

model, weights, arch = load_trained_model("best")    # the shipped checkpoint in models/best_model_weights/
engine = RealWorldInference(model, ImagePreprocessor.for_fashion_mnist(invert=None))
result = engine.predict(np.random.randint(0, 255, (60, 40, 3), dtype=np.uint8), return_top_k=3)
print(arch, result["predicted_class_name"] in ImagePreprocessor.CLASS_NAMES, len(result["top_k_predictions"]))
```

`predict_with_uncertainty` repeats the forward pass with dropout active and
reports the spread; it is only informative for models that contain dropout.

**REST API** (`src/serving/api.py`, FastAPI; interactive docs at `/docs`):

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness and whether a model is loaded |
| POST | `/initialize` | load a model |
| GET | `/model/info` | architecture and parameter count |
| POST | `/predict` | classify one uploaded image |
| POST | `/predict/batch` | classify several images |
| GET | `/predict/uncertainty` | prediction with uncertainty estimate |
| GET | `/status` | server status |

```bash
uvicorn src.serving.api:app --port 8000
```

**Apps and containers.** `apps/gradio_app.py` and `apps/streamlit_dashboard.py`
are interactive demos; `docker/docker-compose.yml` starts the API (port 8000),
Streamlit (8501), Gradio (7860) and Jupyter (8888). See
[`docker/README.md`](../docker/README.md) and [DEPLOYMENT.md](DEPLOYMENT.md).

## Monitoring

`src/monitoring/tracker.py` holds lightweight helpers for a deployed model:
a rolling `MetricsTracker`, a `DriftDetector` comparing incoming inputs with
a reference distribution (KL divergence, Wasserstein distance or a KS test), a
`PredictionMonitor` for class balance and low-confidence predictions, and a
`PerformanceLogger` that writes per-session JSON logs.

```python
import numpy as np
from src.monitoring.tracker import DriftDetector, PredictionMonitor

detector = DriftDetector(reference_data=np.random.rand(500), method="ks_test", threshold=0.05)
drifted, score = detector.detect_drift(np.random.rand(500) + 0.5)
print(bool(drifted))                      # True: the shifted sample is flagged

monitor = PredictionMonitor(num_classes=10)
monitor.update(np.array([1, 2, 2]), np.array([0.9, 0.4, 0.8]))
print(monitor.detect_low_confidence(threshold=0.5))
```

## Configuration

Every setting lives in `config.yaml` and is loaded with
`src.config.settings.load_config`. Any key can also be overridden per run,
without editing the file:

```bash
python src/cli/train.py --model tinyvgg --set augmentation.random_crop=false --set training.label_smoothing=0.1
```

```python
from src.config.settings import load_config

cfg = load_config("config.yaml")
print(cfg.training.optimizer, cfg.get("augmentation.fill"))
cfg.set("training.seed", 7)
print(cfg.get("training.seed"))
```

## Testing

```bash
pytest tests/ -q                   # the full suite; CI runs it on Python 3.10 and 3.11
pytest tests/test_registry.py -v   # one area
pytest tests/ --cov=src            # with coverage
```

The suite covers the models and registry, seeding and metric accumulation,
training, resume and mixed precision, augmentation (including regressions
for both augmentation bugs), calibration and robustness analysis, sweeps and
aggregation, ensembles, inference, the website generator, and every example
on this page.
