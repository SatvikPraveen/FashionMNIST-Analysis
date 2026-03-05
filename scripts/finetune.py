#!/usr/bin/env python3
"""
Hyperparameter tuning script for FashionMNIST-Analysis.

Grid search over learning rates, batch sizes, and early stopping patience.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from itertools import product
from typing import Optional
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.dataset import create_dataloaders
from src.model_definitions import MiniCNN, TinyVGG, ResNet, BasicBlock
from src.utils import get_device, print_device_info
from train import train_model, get_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def finetune_model(
    model_name: str,
    config: Config,
    device: torch.device,
    train_loader,
    val_loader,
    test_loader,
    output_dir: str,
    learning_rates: list,
    batch_sizes: list,
    patience_values: list,
    pretrained_path: Optional[str] = None
):
    """
    Perform grid search hyperparameter tuning.
    
    Args:
        model_name (str): Model architecture name
        config (Config): Base configuration
        device: Training device
        train_loader, val_loader, test_loader: Data loaders
        output_dir (str): Output directory
        learning_rates (list): Learning rates to try
        batch_sizes (list): Batch sizes to try
        patience_values (list): Early stopping patience values
        pretrained_path (str): Path to pretrained weights
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING: {model_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Grid search dimensions:")
    logger.info(f"  Learning rates: {learning_rates}")
    logger.info(f"  Batch sizes: {batch_sizes}")
    logger.info(f"  Patience values: {patience_values}")
    
    total_combinations = len(learning_rates) * len(batch_sizes) * len(patience_values)
    logger.info(f"  Total combinations: {total_combinations}\n")
    
    results = []
    best_val_acc = 0
    best_config = None
    
    for idx, (lr, bs, patience) in enumerate(product(learning_rates, batch_sizes, patience_values), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Configuration {idx}/{total_combinations}")
        logger.info(f"LR: {lr}, BS: {bs}, Patience: {patience}")
        logger.info(f"{'='*60}")
        
        # Update config
        config.set('training.learning_rate', lr)
        config.set('training.batch_size', bs)
        config.set('training.early_stopping_patience', patience)
        
        # Recreate dataloaders with new batch size if needed
        if bs != train_loader.batch_size:
            logger.info(f"Recreating dataloaders with batch size {bs}...")
            new_train_loader, new_val_loader, new_test_loader = create_dataloaders(
                use_torchvision=True,
                batch_size=bs,
                num_workers=0
            )
        else:
            new_train_loader = train_loader
            new_val_loader = val_loader
            new_test_loader = test_loader
        
        # Create model
        model = get_model(model_name, num_classes=10, in_channels=1)
        
        # Load pretrained weights if provided
        if pretrained_path:
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path, map_location=device))
        
        # Train
        history = train_model(
            model=model,
            train_loader=new_train_loader,
            val_loader=new_val_loader,
            test_loader=new_test_loader,
            config=config,
            device=device,
            output_dir=output_dir,
            model_name=f"{model_name}_bs{bs}_lr{lr}_pat{patience}"
        )
        
        # Record results
        result = {
            'learning_rate': lr,
            'batch_size': bs,
            'patience': patience,
            'best_val_acc': max(history['val_acc']),
            'best_val_loss': min(history['val_loss']),
            'test_acc': history.get('test_acc', 0),
            'test_loss': history.get('test_loss', 0),
            'epochs_trained': len(history['train_loss'])
        }
        results.append(result)
        
        # Track best configuration
        if result['best_val_acc'] > best_val_acc:
            best_val_acc = result['best_val_acc']
            best_config = result
        
        logger.info(f"\n📊 Result: Val Acc: {result['best_val_acc']:.4f}, Test Acc: {result['test_acc']:.4f}")
    
    # Save all results
    results_path = os.path.join(output_dir, f"{model_name}_finetuning_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINE-TUNING COMPLETE: {model_name.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"Best configuration:")
    logger.info(f"  LR: {best_config['learning_rate']}")
    logger.info(f"  Batch Size: {best_config['batch_size']}")
    logger.info(f"  Patience: {best_config['patience']}")
    logger.info(f"  Val Acc: {best_config['best_val_acc']:.4f}")
    logger.info(f"  Test Acc: {best_config['test_acc']:.4f}")
    logger.info(f"\nAll results saved to: {results_path}")
    logger.info(f"{'='*60}\n")
    
    return results, best_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FashionMNIST models")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["minicnn", "tinyvgg", "resnet"],
        required=True,
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/fine_tuning_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model weights"
    )
    parser.add_argument(
        "--learning-rates",
        nargs='+',
        type=float,
        default=[1e-5, 5e-6],
        help="Learning rates to try (default: 1e-5 5e-6)"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs='+',
        type=int,
        default=[32, 64],
        help="Batch sizes to try (default: 32 64)"
    )
    parser.add_argument(
        "--patience-values",
        nargs='+',
        type=int,
        default=[2, 3],
        help="Early stopping patience values (default: 2 3)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage"
    )
    
    args = parser.parse_args()
    
    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Get device
    device = print_device_info() if not args.force_cpu else get_device(force_cpu=True)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    logger.info("\n📦 Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        use_torchvision=True,
        batch_size=args.batch_sizes[0],  # Use first batch size initially
        num_workers=0
    )
    
    # Perform fine-tuning
    results, best_config = finetune_model(
        model_name=args.model,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=args.output_dir,
        learning_rates=args.learning_rates,
        batch_sizes=args.batch_sizes,
        patience_values=args.patience_values,
        pretrained_path=args.pretrained
    )
    
    logger.info("🎉 Fine-tuning complete!")


if __name__ == "__main__":
    main()
