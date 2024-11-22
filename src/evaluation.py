import torch
from torch.utils.data import TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CSV to TensorDataset
def load_csv_to_dataset(csv_path):
    """
    Loads data from a CSV file into a PyTorch TensorDataset.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        TensorDataset: A dataset containing features and labels as tensors.
    """

    df = pd.read_csv(csv_path)
    labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)
    features = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
    return TensorDataset(features, labels)

# Evaluate model with confusion matrix
def evaluate_model_with_confusion_matrix(model, dataloader, device):
    """
    Evaluates the model and collects predictions and true labels.

    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader for the dataset to evaluate.
        device: Device for computation (e.g., "cuda", "cpu").

    Returns:
        Tuple[float, float, np.ndarray, np.ndarray]: 
            - Average loss,
            - Accuracy,
            - Array of predictions,
            - Array of true labels.
    """

    model.eval()  # Set model to evaluation mode
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            total_loss += loss.item()
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            total_correct += (y_pred == y).sum().item()
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

# Visualize predictions
def visualize_predictions(model, dataloader,  device, result_dir= "figures/evaluation_plots", class_names=None, filename="prediction_visualization.png"):
    """
    Visualize a few predictions with their true labels in a 3x3 grid and save the plot.

    Args:
        model: The trained model.
        dataloader: DataLoader for the dataset.
        device: The device to compute on.
        result_dir (str): Directory to save the plot.
        class_names (list, optional): Optional list of class names.
        filename (str): Name of the file to save the plot as.

    Saves:
        The visualization plot in the specified directory.
    """

    model.eval()
    X_batch, y_batch = next(iter(dataloader))
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    with torch.inference_mode():
        y_logits = model(X_batch)
        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_batch[i].squeeze().cpu(), cmap="gray")
        title = f"True: {y_batch[i].item()}, Pred: {y_pred[i].item()}"
        if class_names:
            title = f"True: {class_names[y_batch[i].item()]}, Pred: {class_names[y_pred[i].item()]}"
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout(pad=2.0)
    save_path = os.path.join(result_dir, os.path.basename(filename))
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✅ Prediction visualization saved at: {save_path}")
    plt.show()

# Function to generate and save confusion matrix
def save_confusion_matrix(true_labels, predictions, class_names, result_dir="tests", filename="confusion_matrix.png"):
    """
    Generates and saves a confusion matrix plot.

    Args:
        true_labels: Ground truth labels.
        predictions: Predicted labels.
        class_names: List of class names for the labels.
        result_dir: Directory to save the plot.
        filename: Filename for the confusion matrix plot.
    """
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap="Blues", xticks_rotation="vertical", ax=plt.gca())
    plt.title("Confusion Matrix", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save confusion matrix plot
    save_path = os.path.join(result_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✅ Confusion matrix saved at: {save_path}")
    plt.close()

# Function to calculate and save metrics
def save_metrics(true_labels, predictions, result_dir="tests", filename="evaluation_metrics.csv"):
    """
    Calculates and saves evaluation metrics (precision, recall, F1-score, accuracy) into a CSV file.

    Args:
        true_labels: Ground truth labels.
        predictions: Predicted labels.
        result_dir: Directory to save the metrics.
        filename: Filename for the metrics CSV file.
    """
    # Calculate metrics
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")
    accuracy = accuracy_score(true_labels, predictions)

    # Create a DataFrame for metrics
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    # Save metrics to CSV
    save_path = os.path.join(result_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_df.to_csv(save_path, index=False)
    print(f"✅ Metrics saved at: {save_path}")