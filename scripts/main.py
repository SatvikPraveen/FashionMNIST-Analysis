import os
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from src.evaluation import (
    load_csv_to_dataset,
    evaluate_model_with_confusion_matrix,
    visualize_predictions,
    save_confusion_matrix,
    save_metrics
)
from src.model_definitions import ResNet, BasicBlock

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load the pre-trained model
def load_model(model_path, model_name="ResNet", num_classes=10):
    """
    Load a pre-trained model and its weights.

    Args:
        model_path (str): Path to the pre-trained model weights.
        model_name (str): Name of the model architecture (default: "ResNet").
        num_classes (int): Number of output classes (default: 10).

    Returns:
        torch.nn.Module: Loaded model set to evaluation mode.
    """
    if model_name == "ResNet":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
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
    parser.add_argument('--test_csv', type=str, required=True, help="Path to the test dataset CSV.")
    parser.add_argument('--test_dir', type=str, default="tests", help="Directory to store all results (CSV, images, etc.).")
    args = parser.parse_args()

    # Create the tests directory
    os.makedirs(args.test_dir, exist_ok=True)

    # Load the test data
    print("🔄 Loading test data...")
    test_data = load_csv_to_dataset(args.test_csv)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    print(f"✅ Test data loaded: {len(test_data)} samples.")

    # Load the model
    model = load_model(args.model_path, model_name="ResNet")

    # Evaluate the model and collect predictions
    test_loss, test_accuracy, predictions, true_labels = evaluate_model_with_confusion_matrix(
        model, test_loader, device
    )
    print(f"\n🎯 Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Save predictions as a CSV file inside `tests`
    predictions_csv_path = os.path.join(args.test_dir, "predictions_vector.csv")
    pd.DataFrame({"True Labels": true_labels, "Predicted Labels": predictions}).to_csv(predictions_csv_path, index=False)
    print(f"✅ Predictions saved to {predictions_csv_path}")

    # Visualize confusion matrix and predictions
    print("\n🔄 Generating and saving confusion matrix and predictions...")
    confusion_matrix_path = os.path.join(args.test_dir, "Best_ResNet_confusion_matrix.png")
    cm = confusion_matrix(true_labels, predictions)

    # Save Visualized Predictions
    visualize_predictions(
        model, test_loader, device, result_dir= args.test_dir,filename=os.path.join(args.test_dir, "prediction_visualization.png")
    )
    print(f"✅ Prediction visualization successfully saved in {args.test_dir}.")
    # Save confusion matrix and metrics
    print("\n🔄 Generating and saving confusion matrix and metrics...")
    class_names = [str(i) for i in range(10)]  # Class names for Fashion MNIST (0-9)
    save_confusion_matrix(true_labels, predictions, class_names, result_dir=args.test_dir)
    save_metrics(true_labels, predictions, result_dir=args.test_dir)
    print(f"✅ All results saved in {args.test_dir}.")


if __name__ == "__main__":
    main()