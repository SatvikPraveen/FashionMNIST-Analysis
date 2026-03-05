"""
Transfer Learning module for FashionMNIST-Analysis.

This module provides utilities for loading and fine-tuning pretrained models
including Vision Transformers, EfficientNet, and other SOTA architectures.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TransferLearningModel:
    """
    Wrapper for transfer learning models with pretrained weights.
    
    Supports Vision Transformers, EfficientNet, and other TIMM models.
    """
    
    def __init__(self, model_name: str = "vit_base_patch16_224", num_classes: int = 10, 
                 pretrained: bool = True, device: str = "auto"):
        """
        Initialize transfer learning model.
        
        Args:
            model_name (str): Model name from TIMM library
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            device (str): Device to load model on (auto, cuda, cpu, mps)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        
    def _load_model(self) -> nn.Module:
        """
        Load pretrained model from TIMM.
        
        Returns:
            nn.Module: Loaded model
        """
        logger.info(f"Loading {self.model_name} with pretrained={self.pretrained}")
        
        # Load model from TIMM
        model = timm.create_model(self.model_name, pretrained=self.pretrained, 
                                   num_classes=self.num_classes)
        
        # Handle grayscale to RGB conversion if needed
        if "vit" in self.model_name.lower() or "efficientnet" in self.model_name.lower():
            # These models expect RGB input, so we need to adapt for grayscale Fashion MNIST
            model = self._adapt_for_grayscale(model)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully: {self.model_name}")
        return model
    
    def _adapt_for_grayscale(self, model: nn.Module) -> nn.Module:
        """
        Adapt pretrained RGB models to accept grayscale input.
        
        Converts first conv layer to accept 1 channel input and averages weights.
        
        Args:
            model (nn.Module): Pretrained model
            
        Returns:
            nn.Module: Adapted model
        """
        # Get first layer
        if hasattr(model, 'patch_embed'):  # ViT
            first_conv = model.patch_embed.proj
        elif hasattr(model, 'conv1'):  # ResNet-like or EfficientNet
            first_conv = model.conv1
        else:
            logger.warning("Could not find first conv layer for adaptation")
            return model
        
        # Create new layer for grayscale
        in_channels = first_conv.in_channels
        if in_channels == 3:
            old_weight = first_conv.weight.data
            # Average across RGB channels
            new_weight = old_weight.mean(dim=1, keepdim=True)
            
            # Create new conv layer
            new_conv = nn.Conv2d(1, first_conv.out_channels, 
                                first_conv.kernel_size, 
                                first_conv.stride,
                                first_conv.padding, 
                                first_conv.bias is not None)
            new_conv.weight.data = new_weight
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data
            
            # Replace first layer
            if hasattr(model, 'patch_embed'):
                model.patch_embed.proj = new_conv
            elif hasattr(model, 'conv1'):
                model.conv1 = new_conv
            
            logger.info("Adapted model for grayscale input (1 channel -> 1 channel)")
        
        return model
    
    def freeze_backbone(self, num_layers: int = 0) -> None:
        """
        Freeze backbone layers for transfer learning.
        
        Args:
            num_layers (int): Number of layers to freeze (0 = freeze all)
        """
        if num_layers == 0:
            # Freeze all except classifier
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            if hasattr(self.model, 'head'):
                for param in self.model.head.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, 'fc'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            logger.info("Froze entire backbone")
        else:
            logger.info(f"Freezing first {num_layers} layers")
            layer_count = 0
            for param in self.model.parameters():
                param.requires_grad = False
                layer_count += 1
                if layer_count >= num_layers:
                    break
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Unfroze all parameters")
    
    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        return self.model
    
    def to_device(self, device: str) -> None:
        """Move model to device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        logger.info(f"Model moved to {self.device}")


def load_vit_model(num_classes: int = 10, pretrained: bool = True, 
                   device: str = "auto") -> nn.Module:
    """
    Load Vision Transformer model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        device (str): Device to load on
        
    Returns:
        nn.Module: Vision Transformer model
    """
    tl_model = TransferLearningModel("vit_base_patch16_224", num_classes, pretrained, device)
    return tl_model.model


def load_efficientnet_model(num_classes: int = 10, pretrained: bool = True,
                            device: str = "auto", version: str = "b0") -> nn.Module:
    """
    Load EfficientNet model.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        device (str): Device to load on
        version (str): EfficientNet version (b0-b7)
        
    Returns:
        nn.Module: EfficientNet model
    """
    model_name = f"efficientnet_{version}"
    tl_model = TransferLearningModel(model_name, num_classes, pretrained, device)
    return tl_model.model


def load_resnet50_model(num_classes: int = 10, pretrained: bool = True,
                        device: str = "auto") -> nn.Module:
    """
    Load ResNet50 model (deeper than baseline ResNet18).
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        device (str): Device to load on
        
    Returns:
        nn.Module: ResNet50 model
    """
    tl_model = TransferLearningModel("resnet50", num_classes, pretrained, device)
    return tl_model.model


def get_available_models() -> list:
    """
    Get list of available models in TIMM.
    
    Filters for commonly used vision models.
    
    Returns:
        list: List of available model names
    """
    all_models = timm.list_models()
    # Filter for commonly used models
    preferred = [m for m in all_models if any(x in m for x in 
                 ['vit', 'efficientnet', 'resnet', 'convnext', 'densenet'])]
    return preferred[:50]  # Return first 50


def print_model_info(model: nn.Module) -> None:
    """
    Print model information including parameter count.
    
    Args:
        model (nn.Module): Model to inspect
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Frozen parameters: {total_params - trainable_params:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    logger.info("Loading Vision Transformer...")
    vit = load_vit_model(num_classes=10, pretrained=True)
    print_model_info(vit)
    
    logger.info("\nLoading EfficientNet-B0...")
    eff = load_efficientnet_model(num_classes=10, pretrained=True)
    print_model_info(eff)
