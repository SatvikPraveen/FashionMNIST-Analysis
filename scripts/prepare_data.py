#!/usr/bin/env python3
"""
Data preparation script for FashionMNIST-Analysis.

Automates downloading and preparing Fashion MNIST dataset for training.
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from torchvision import datasets
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_fashion_mnist(data_dir: str = "./data") -> tuple:
    """
    Download Fashion MNIST dataset using torchvision.
    
    Args:
        data_dir (str): Directory to save data
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    logger.info(f"Downloading Fashion MNIST to {data_dir}...")
    
    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True
    )
    
    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True
    )
    
    logger.info(f"✅ Downloaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_dataset, test_dataset


def dataset_to_csv(dataset, output_path: str):
    """
    Convert torchvision dataset to CSV format.
    
    Args:
        dataset: Torchvision dataset
        output_path (str): Path to save CSV file
    """
    logger.info(f"Converting dataset to CSV: {output_path}")
    
    # Extract data
    images = []
    labels = []
    
    for img, label in dataset:
        # Convert PIL image to numpy array and flatten
        img_array = np.array(img).flatten()
        images.append(img_array)
        labels.append(label)
    
    # Create DataFrame
    images_array = np.array(images)
    df = pd.DataFrame(images_array)
    df.insert(0, 'label', labels)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved: {output_path} ({len(df)} samples, {df.shape[1]} columns)")


def split_train_val(train_dataset, train_split: float = 0.8) -> tuple:
    """
    Split training dataset into train and validation sets.
    
    Args:
        train_dataset: Full training dataset
        train_split (float): Fraction for training (rest for validation)
        
    Returns:
        Tuple of (train_data, val_data)
    """
    total_samples = len(train_dataset)
    train_size = int(total_samples * train_split)
    
    # Get all data
    all_images = []
    all_labels = []
    
    for img, label in train_dataset:
        all_images.append(np.array(img).flatten())
        all_labels.append(label)
    
    # Shuffle
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Split data
    train_images = [all_images[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_images = [all_images[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    logger.info(f"Split: {len(train_images)} train, {len(val_images)} validation samples")
    
    return (train_images, train_labels), (val_images, val_labels)


def save_split_to_csv(images, labels, output_path: str):
    """Save images and labels to CSV."""
    df = pd.DataFrame(images)
    df.insert(0, 'label', labels)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved: {output_path}")


def prepare_data(
    data_dir: str = "./data",
    output_dir: str = "./data_preparation",
    train_split: float = 0.8,
    save_csv: bool = True,
    random_seed: int = 42
):
    """
    Complete data preparation pipeline.
    
    Args:
        data_dir (str): Directory for raw data
        output_dir (str): Directory for processed CSVs
        train_split (float): Train/val split ratio
        save_csv (bool): Save as CSV files
        random_seed (int): Random seed for reproducibility
    """
    logger.info("\n" + "="*60)
    logger.info("FASHION MNIST DATA PREPARATION")
    logger.info("="*60)
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Download data
    train_dataset, test_dataset = download_fashion_mnist(data_dir)
    
    if save_csv:
        # Split train into train/val
        (train_images, train_labels), (val_images, val_labels) = split_train_val(
            train_dataset, train_split
        )
        
        # Save to CSV
        train_csv = os.path.join(output_dir, "fashion_mnist_train.csv")
        val_csv = os.path.join(output_dir, "fashion_mnist_val.csv")
        test_csv = os.path.join(output_dir, "fashion_mnist_test.csv")
        
        save_split_to_csv(train_images, train_labels, train_csv)
        save_split_to_csv(val_images, val_labels, val_csv)
        
        # Save test set
        test_images = [np.array(img).flatten() for img, _ in test_dataset]
        test_labels = [label for _, label in test_dataset]
        save_split_to_csv(test_images, test_labels, test_csv)
        
        logger.info(f"\n📁 CSV files saved to: {output_dir}")
        logger.info(f"   - Train: {train_csv}")
        logger.info(f"   - Val:   {val_csv}")
        logger.info(f"   - Test:  {test_csv}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ DATA PREPARATION COMPLETE")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Fashion MNIST dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for raw data (default: ./data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data_preparation",
        help="Directory for processed CSV files (default: ./data_preparation)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV conversion (only download)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    prepare_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        save_csv=not args.no_csv,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
