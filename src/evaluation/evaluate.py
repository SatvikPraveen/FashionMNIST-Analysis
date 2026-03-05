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

# Auto-select best available device (CUDA > MPS > CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load the pre-trained model
def load_model(model_path, model_name="ResNet", num_classes=10):
    """
    Load a pre-trained model and its weights.

    Args:
        model_path (str): Path to the pre-trained model weights.
        model_name (str): Name of the model architecture ("ResNet", "TinyVGG", or "MiniCNN").
        num_classes (int): Number of output classes (default: 10).

    Returns:
        torch.nn.Module: Loaded model set to evaluation mode.
    """
    name = model_name.lower()
    if name == "resnet":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    elif name == "tinyvgg":
        model = TinyVGG(in_channels=1, hidden_units=64, num_classes=num_classes)
    elif name == "minicnn":
        model = MiniCNN(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Choose from: ResNet, TinyVGG, MiniCNN")
    
    print(f"🔄 Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only= True))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    return model

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
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
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
    class_names = [str(i) for i in range(10)]  # Class names for Fashion MNIST (0-9)
    save_confusion_matrix(true_labels, predictions, class_names, result_dir=args.figures_dir)
    save_metrics(true_labels, predictions, result_dir=args.results_dir)
    print(f"✅ Plots saved in {args.figures_dir}.")
    print(f"✅ CSVs saved in {args.results_dir}.")


if __name__ == "__main__":
    main()