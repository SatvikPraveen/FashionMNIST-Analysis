"""
Dataset classes for FashionMNIST-Analysis.

Provides unified dataset interface with augmentation support.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Callable
from torchvision import datasets, transforms
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FashionMNISTDataset(Dataset):
    """
    Custom Dataset for Fashion MNIST from CSV files.
    
    Supports data augmentation through transform pipeline.
    """
    
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Initialize dataset from CSV file.
        
        Args:
            csv_path (str): Path to CSV file with pixel data
            transform (callable): Optional transform to apply to images
            target_transform (callable): Optional transform for labels
        """
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.target_transform = target_transform
        
        # Extract labels and pixel data
        if 'label' in self.data.columns:
            self.labels = self.data['label'].values
            self.pixels = self.data.drop('label', axis=1).values
        else:
            # Assume first column is label
            self.labels = self.data.iloc[:, 0].values
            self.pixels = self.data.iloc[:, 1:].values
        
        logger.info(f"Loaded dataset from {csv_path}: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get pixel values and reshape to 28x28
        pixels = self.pixels[idx].reshape(28, 28).astype(np.float32) / 255.0
        label = int(self.labels[idx])
        
        # Convert to tensor
        image = torch.from_numpy(pixels).unsqueeze(0)  # Add channel dimension (1, 28, 28)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class FashionMNISTFromTorchvision(Dataset):
    """
    Wrapper for torchvision FashionMNIST dataset.
    
    Provides consistent interface with CSV-based dataset.
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize from torchvision dataset.
        
        Args:
            root (str): Root directory for data
            train (bool): Load training set if True, test set if False
            download (bool): Download dataset if not present
            transform (callable): Optional transform
        """
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()  # Base transform
        )
        self.transform = transform
        
        logger.info(f"Loaded FashionMNIST {'train' if train else 'test'} set: {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_default_transforms(image_size: int = 28, augment: bool = False) -> transforms.Compose:
    """
    Get default transforms for FashionMNIST.
    
    Args:
        image_size (int): Target image size
        augment (bool): Whether to include augmentation
        
    Returns:
        Composed transforms
    """
    transform_list = []
    
    if augment:
        transform_list.extend([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    
    if image_size != 28:
        transform_list.append(transforms.Resize((image_size, image_size)))
    
    # Normalization for Fashion MNIST (dataset statistics: mean=0.2860, std=0.3530)
    transform_list.extend([
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_torchvision: bool = False,
    data_root: str = "./data"
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_csv (str): Path to training CSV (if not using torchvision)
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        train_transform (callable): Transform for training data
        val_transform (callable): Transform for val/test data
        use_torchvision (bool): Use torchvision dataset instead of CSV
        data_root (str): Root directory for torchvision data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader, val_loader, test_loader = None, None, None
    
    # Default transforms if not provided
    if train_transform is None:
        train_transform = get_default_transforms(augment=False)
    if val_transform is None:
        val_transform = get_default_transforms(augment=False)
    
    if use_torchvision:
        # Use torchvision datasets
        train_dataset = FashionMNISTFromTorchvision(
            root=data_root,
            train=True,
            download=True,
            transform=train_transform
        )
        test_dataset = FashionMNISTFromTorchvision(
            root=data_root,
            train=False,
            download=True,
            transform=val_transform
        )
        
        # Split train into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
    else:
        # Use CSV datasets
        if train_csv:
            train_dataset = FashionMNISTDataset(train_csv, transform=train_transform)
        else:
            train_dataset = None
        
        if val_csv:
            val_dataset = FashionMNISTDataset(val_csv, transform=val_transform)
        else:
            val_dataset = None
        
        if test_csv:
            test_dataset = FashionMNISTDataset(test_csv, transform=val_transform)
        else:
            test_dataset = None
    
    # Create dataloaders
    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Train loader created: {len(train_loader)} batches")
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Validation loader created: {len(val_loader)} batches")
    
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Test loader created: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


# Fashion MNIST class names
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    print("\n" + "="*60)
    print("DATASET MODULE TEST")
    print("="*60)
    
    # Test torchvision dataset
    dataset = FashionMNISTFromTorchvision(download=True)
    print(f"\n✅ Torchvision dataset loaded: {len(dataset)} samples")
    
    # Test single item
    image, label = dataset[0]
    print(f"   Image shape: {image.shape}")
    print(f"   Label: {label} ({FASHION_MNIST_CLASSES[label]})")
    
    # Test dataloader creation
    train_loader, val_loader, test_loader = create_dataloaders(
        use_torchvision=True,
        batch_size=32
    )
    print(f"\n✅ Dataloaders created")
    print(f"   Train batches: {len(train_loader) if train_loader else 0}")
    print(f"   Val batches: {len(val_loader) if val_loader else 0}")
    print(f"   Test batches: {len(test_loader) if test_loader else 0}")
    
    print("="*60 + "\n")
