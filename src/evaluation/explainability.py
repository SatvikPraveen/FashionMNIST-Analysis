"""
Explainability module for FashionMNIST-Analysis.

Provides visualization tools for model interpretability including:
- Grad-CAM: Class Activation Maps
- Attention Maps: For Vision Transformers
- Feature importance and gradient-based explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — prevents plt.show() from blocking
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Visualizes which regions of the input image are important for the model's prediction.
    """
    
    def __init__(self, model: nn.Module, target_layer: str, device: str = "cpu"):
        """
        Initialize Grad-CAM.
        
        Args:
            model (nn.Module): Target model
            target_layer (str): Name of target layer for gradient computation
            device (str): Device to compute on
        """
        self.model = model
        self.device = torch.device(device)
        self.target_layer = target_layer
        
        # Register hooks
        self.gradients = None
        self.activations = None
        self._register_hooks()
        
        logger.info(f"Grad-CAM initialized for layer: {target_layer}")
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                logger.info(f"Hooks registered on layer: {name}")
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor (B, C, H, W)
            class_idx (int, optional): Target class index. If None, uses predicted class
            
        Returns:
            np.ndarray: CAM heatmap
        """
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        
        # Compute CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Weight by average gradient
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], device=self.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize_cam(self, image: np.ndarray, cam: np.ndarray, 
                     class_name: str = "", save_path: Optional[str] = None) -> None:
        """
        Visualize CAM overlay on image.
        
        Args:
            image (np.ndarray): Input image
            cam (np.ndarray): CAM heatmap
            class_name (str): Class name for title
            save_path (str, optional): Path to save figure
        """
        # Resize CAM to match image
        cam_resized = np.array(plt.imshow(cam, cmap='jet').get_figure().canvas.buffer_rgba())
        cam_resized = np.array(plt.imread(save_path) if save_path else cam)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # CAM
        im = axes[1].imshow(cam, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[2].imshow(cam, cmap='jet', alpha=0.5)
        axes[2].set_title(f"Overlay - {class_name}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Grad-CAM visualization saved: {save_path}")
        
        plt.close()


class AttentionMapper:
    """
    Extract and visualize attention maps from Vision Transformer models.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize attention mapper.
        
        Args:
            model (nn.Module): Vision Transformer model
            device (str): Device to compute on
        """
        self.model = model
        self.device = torch.device(device)
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # For multi-head attention, output is (B, num_heads, seq_len, seq_len)
            if isinstance(output, tuple):
                self.attention_maps.append(output[0].detach())
            else:
                self.attention_maps.append(output.detach())
        
        # Register hooks on attention modules
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                module.register_forward_hook(attention_hook)
    
    def get_attention_maps(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention maps for input.
        
        Args:
            input_tensor (torch.Tensor): Input tensor
            
        Returns:
            List[torch.Tensor]: Attention maps from each layer
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(input_tensor)
        
        return self.attention_maps
    
    def visualize_attention(self, image: np.ndarray, attention_map: np.ndarray,
                           save_path: Optional[str] = None) -> None:
        """
        Visualize attention map overlay on image.
        
        Args:
            image (np.ndarray): Input image
            attention_map (np.ndarray): Attention map
            save_path (str, optional): Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        im = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title("Attention Map")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Attention visualization saved: {save_path}")
        
        plt.close()


def get_layer_activations(model: nn.Module, input_tensor: torch.Tensor,
                         layer_name: str, device: str = "cpu") -> np.ndarray:
    """
    Extract activations from a specific layer.
    
    Args:
        model (nn.Module): Model
        input_tensor (torch.Tensor): Input tensor
        layer_name (str): Name of target layer
        device (str): Device to compute on
        
    Returns:
        np.ndarray: Layer activations
    """
    activations = None
    
    def hook_fn(module, input, output):
        nonlocal activations
        activations = output.detach().cpu().numpy()
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook_fn)
            break
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    
    return activations


def visualize_filters(layer_weights: torch.Tensor, num_filters: int = 16,
                     save_path: Optional[str] = None) -> None:
    """
    Visualize convolutional filters.
    
    Args:
        layer_weights (torch.Tensor): Layer weights (out_channels, in_channels, H, W)
        num_filters (int): Number of filters to display
        save_path (str, optional): Path to save figure
    """
    weights = layer_weights.cpu().detach().numpy()
    
    # Normalize weights
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    # Handle different filter sizes
    if len(weights.shape) == 4:
        filters = weights[:min(num_filters, weights.shape[0])]
        
        # For RGB filters
        if filters.shape[1] == 3:
            filters = filters.transpose(0, 2, 3, 1)
        else:
            filters = filters.mean(axis=1)
    else:
        logger.warning("Unexpected filter shape")
        return
    
    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(filters))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(filters):
            ax.imshow(filters[idx], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Filter visualization saved: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Explainability module loaded successfully")
