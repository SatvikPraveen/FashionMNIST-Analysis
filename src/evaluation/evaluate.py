"""
Standalone model evaluation script for FashionMNIST-Analysis.

Loads a pre-trained model (MiniCNN, TinyVGG, or ResNet) from disk, runs it over a test CSV, and saves:
    - predictions_vector.csv  (true vs. predicted labels)
    - confusion matrix figure
    - evaluation_metrics.csv  (accuracy, precision, recall, F1)
    - prediction visualisation grid

CLI usage:
    python src/cli/evaluate.py \\
        --model_path models/best_model_weights/best_model_weights.pth \\
        --test_csv   data/processed/test.csv \\
        --test_dir   tests/

    # Or run directly:
    python -m src.evaluation.evaluate --model_path ... --test_csv ...
"""

import os
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from src.evaluation.metrics import (
    load_csv_to_dataset,
    evaluate_model_with_confusion_matrix,
    visualize_predictions,
    save_confusion_matrix,
    save_metrics
)
import json
from src.models.architectures import ResNet, BasicBlock, MiniCNN, TinyVGG
from src.models.registry import ModelSpec, build_from_spec, spec_path_for, resolve_name
from src.evaluation.analysis import run_full_analysis, summarize_report, CLASS_NAMES

# Auto-select best available device (CUDA > MPS > CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load the pre-trained model
def load_model(model_path, model_name="ResNet", num_classes=10):
    """
    Load a trained model and its weights.

    If ``<model_path minus .pth>_spec.json`` exists (written by train.py) the
    architecture is rebuilt from it, so any registry model (custom CNN or
    timm backbone) can be evaluated. Otherwise ``model_name`` must be one of
    the custom CNNs.

    Args:
        model_path (str): Path to the model weights.
        model_name (str): Architecture name used when no spec file exists.
        num_classes (int): Number of output classes (default: 10).

    Returns:
        torch.nn.Module: Loaded model set to evaluation mode.
    """
    spec_path = spec_path_for(model_path)
    if os.path.exists(spec_path):
        spec = ModelSpec.load(spec_path)
        spec.pretrained = False  # weights come from the checkpoint
        print(f"[INFO] Rebuilding '{spec.name}' from {spec_path}")
    else:
        name = resolve_name(model_name)
        if name not in ("resnet", "tinyvgg", "minicnn"):
            raise ValueError(
                f"No spec file at {spec_path}; without it only the custom CNNs "
                f"(ResNet, TinyVGG, MiniCNN) can be rebuilt, got '{model_name}'")
        spec = ModelSpec(name=name, num_classes=num_classes)
    model = build_from_spec(spec)

    print(f"🔄 Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    return model


def model_seed(model_path):
    """The training seed of a checkpoint, from run.json or best_model_info.json, else None."""
    d = os.path.dirname(os.path.abspath(model_path))
    for name, key in (("run.json", ("params", "seed")), ("best_model_info.json", ("seed",))):
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path) as f:
                value = json.load(f)
            for k in key:
                value = value.get(k) if isinstance(value, dict) else None
            if value is not None:
                return int(value)
    return None


def csv_split_seed(val_csv):
    """The seed recorded by prepare_data in split.json next to the CSVs, else None."""
    path = os.path.join(os.path.dirname(os.path.abspath(val_csv)), "split.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("seed")
    return None


def resolve_val_loader(model_path, val_csv, batch_size):
    """
    A validation loader for temperature scaling that the model did not train on.

    The validation CSV is only held out for a model trained with the seed
    that produced it (recorded in split.json). When the model's seed is known
    and differs, its own validation split is rebuilt from that seed instead.
    """
    seed = model_seed(model_path)
    csv_seed = csv_split_seed(val_csv) if val_csv else None
    if val_csv and seed is not None and csv_seed == seed:
        print(f"[INFO] Temperature scaling on {val_csv} (split seed {csv_seed} matches the model).")
        return DataLoader(load_csv_to_dataset(val_csv), batch_size=batch_size, shuffle=False)
    if seed is not None:
        from src.data.dataset import create_dataloaders
        print(f"[INFO] Temperature scaling on the model's own validation split (seed {seed}).")
        _, val_loader, _ = create_dataloaders(use_torchvision=True, batch_size=batch_size, seed=seed)
        return val_loader
    if val_csv:
        print(f"[WARN] The model's training seed is unknown, so {val_csv} may overlap its training "
              f"data; the fitted temperature can be optimistic.")
        return DataLoader(load_csv_to_dataset(val_csv), batch_size=batch_size, shuffle=False)
    print("[INFO] No validation data: skipping temperature scaling.")
    return None


# Main function
def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained Fashion MNIST models.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the pre-trained model weights.")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Architecture name: ResNet, TinyVGG, MiniCNN. Auto-detected from best_model_info.json if omitted.")
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument('--figures_dir', type=str, default="figures/evaluation_plots",
                        help="Directory to save plots (confusion matrix, prediction grid).")
    parser.add_argument('--results_dir', type=str, default="results/evaluation_results",
                        help="Directory to save CSV outputs (predictions, metrics).")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--analysis', action='store_true',
                        help="Also run calibration / per-class / robustness analysis "
                             "(writes analysis.json + figures).")
    parser.add_argument('--val_csv', type=str, default=None,
                        help="Validation CSV for temperature scaling (with --analysis). Used only when "
                             "its split seed matches the model's; otherwise the model's own validation "
                             "split is rebuilt from its seed.")
    parser.add_argument('--no_robustness', action='store_true',
                        help="Skip the corruption sweep in --analysis (7 corruptions x 5 severities).")
    args = parser.parse_args()

    # Auto-detect architecture from best_model_info.json if --model_name not given
    if args.model_name is None:
        info_path = os.path.join(os.path.dirname(args.model_path), "best_model_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            args.model_name = info["model_name"]
            print(f"[INFO] Auto-detected model architecture from {info_path}: {args.model_name}")
        else:
            args.model_name = "ResNet"
            print(f"[INFO] No --model_name given and no best_model_info.json found; defaulting to ResNet.")

    # Create output directories
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Load the test data
    print("🔄 Loading test data...")
    test_data = load_csv_to_dataset(args.test_csv)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print(f"✅ Test data loaded: {len(test_data)} samples.")

    # Load the model
    model = load_model(args.model_path, model_name=args.model_name)

    # Evaluate the model and collect predictions
    test_loss, test_accuracy, predictions, true_labels = evaluate_model_with_confusion_matrix(
        model, test_loader, device
    )
    print(f"\n🎯 Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Save predictions CSV
    predictions_csv_path = os.path.join(args.results_dir, "predictions_vector.csv")
    pd.DataFrame({"True Labels": true_labels, "Predicted Labels": predictions}).to_csv(predictions_csv_path, index=False)
    print(f"✅ Predictions saved to {predictions_csv_path}")

    # Visualize confusion matrix and predictions
    print("\n🔄 Generating and saving confusion matrix and predictions...")
    confusion_matrix_path = os.path.join(args.figures_dir, f"Best_{args.model_name}_confusion_matrix.png")
    cm = confusion_matrix(true_labels, predictions)

    # Save Visualized Predictions
    visualize_predictions(
        model, test_loader, device, result_dir=args.figures_dir, filename="prediction_visualization.png"
    )
    print(f"✅ Prediction visualization successfully saved in {args.figures_dir}.")
    # Save confusion matrix and metrics
    print("\n🔄 Generating and saving confusion matrix and metrics...")
    class_names = list(CLASS_NAMES)
    save_confusion_matrix(true_labels, predictions, class_names, result_dir=args.figures_dir)
    save_metrics(true_labels, predictions, result_dir=args.results_dir)
    print(f"✅ Plots saved in {args.figures_dir}.")
    print(f"✅ CSVs saved in {args.results_dir}.")

    if args.analysis:
        print("\n🔬 Running calibration / per-class / robustness analysis...")
        val_loader = resolve_val_loader(args.model_path, args.val_csv, args.batch_size)
        report = run_full_analysis(
            model, test_loader, device, out_dir=args.results_dir, val_loader=val_loader,
            class_names=class_names, robustness=not args.no_robustness,
            figures_dir=args.figures_dir,
        )
        print(summarize_report(report))
        print(f"✅ analysis.json written to {args.results_dir}")


if __name__ == "__main__":
    main()