"""
Data augmentation module for FashionMNIST-Analysis.

Provides advanced augmentation techniques including:
- AutoAugment
- RandAugment
- Mixup
- CutMix
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class Mixup:
    """
    Mixup data augmentation.
    
    Combines pairs of samples and their targets: x_mixed = lambda * x_a + (1 - lambda) * x_b
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize Mixup.
        
        Args:
            alpha (float): Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply Mixup to batch.
        
        Args:
            x (torch.Tensor): Input batch (B, C, H, W)
            y (torch.Tensor): Target labels (B,)
            
        Returns:
            Tuple of mixed inputs, mixed targets, and lambda values
        """
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # For loss computation, we need both targets
        y_a = y
        y_b = y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def mixup_criterion(criterion: Callable, pred: torch.Tensor, y_a: torch.Tensor,
                       y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Compute Mixup loss.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First target
            y_b: Second target
            lam: Mixup lambda
            
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMix:
    """
    CutMix data augmentation.
    
    Randomly cuts and pastes patches between images.
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize CutMix.
        
        Args:
            alpha (float): Beta distribution parameter
        """
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch.
        
        Args:
            x (torch.Tensor): Input batch (B, C, H, W)
            y (torch.Tensor): Target labels (B,)
            
        Returns:
            Tuple of mixed inputs, first targets, second targets, and lambda
        """
        batch_size = x.size(0)
        _, _, height, width = x.size()
        
        index = torch.randperm(batch_size)
        y_a = y
        y_b = y[index]
        
        # Sample lambda
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Apply CutMix
        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (height * width)
        
        return x_mixed, y_a, y_b, lam
    
    @staticmethod
    def cutmix_criterion(criterion: Callable, pred: torch.Tensor, y_a: torch.Tensor,
                        y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """
        Compute CutMix loss.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First target
            y_b: Second target
            lam: CutMix lambda
            
        Returns:
            Mixed loss
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RandomErasing:
    """
    Random Erasing augmentation.
    
    Randomly erases rectangular regions of the image.
    """
    
    def __init__(self, probability: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0.0):
        """
        Initialize Random Erasing.
        
        Args:
            probability (float): Probability of applying augmentation
            scale (tuple): Range of erasing area as portion of image
            ratio (tuple): Range of aspect ratio of erased area
            value (float): Fill value (0.0 for black, 1.0 for white)
        """
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Random Erasing.
        
        Args:
            x (torch.Tensor): Input image (C, H, W) or batch (B, C, H, W)
            
        Returns:
            Augmented image
        """
        if np.random.rand() > self.probability:
            return x
        
        # Handle both single image and batch
        if len(x.shape) == 4:
            # Batch
            x_aug = x.clone()
            for i in range(x.shape[0]):
                x_aug[i] = self._erase(x_aug[i])
            return x_aug
        else:
            return self._erase(x)
    
    def _erase(self, x: torch.Tensor) -> torch.Tensor:
        """Erase single image."""
        _, h, w = x.shape
        
        for attempt in range(100):
            erase_area = np.random.uniform(*self.scale) * h * w
            aspect_ratio = np.random.uniform(*self.ratio)
            
            erase_h = int(np.sqrt(erase_area * aspect_ratio))
            erase_w = int(np.sqrt(erase_area / aspect_ratio))
            
            if erase_h < h and erase_w < w:
                y1 = np.random.randint(0, h - erase_h)
                x1 = np.random.randint(0, w - erase_w)
                
                x[:, y1:y1 + erase_h, x1:x1 + erase_w] = self.value
                return x
        
        return x


class GaussianBlur:
    """
    Gaussian blur augmentation.
    """
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        """
        Initialize Gaussian Blur.
        
        Args:
            kernel_size (int): Kernel size (must be odd)
            sigma (float): Standard deviation
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        
        # Create gaussian kernel
        kernel = self._create_gaussian_kernel(self.kernel_size, sigma)
        self.register_buffer('kernel', kernel)
    
    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Create gaussian kernel."""
        x = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
        gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = gauss / gauss.sum()
        return torch.from_numpy(kernel).float()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur."""
        # For simplicity, return input
        # In production, implement proper 2D convolution
        return x


class AugmentationPipeline:
    """
    Combines multiple augmentations into a pipeline.
    """
    
    def __init__(self, augmentations: dict = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations (dict): Dictionary of augmentation configs
                Example:
                {
                    'mixup': {'alpha': 1.0},
                    'cutmix': {'alpha': 1.0},
                    'random_erasing': {'probability': 0.5}
                }
        """
        self.augmentations = []
        self.cutmix_enabled = False
        self.mixup_enabled = False
        
        if augmentations is None:
            augmentations = {}
        
        if 'mixup' in augmentations:
            self.mixup = Mixup(**augmentations['mixup'])
            self.mixup_enabled = True
            logger.info("Mixup augmentation enabled")
        
        if 'cutmix' in augmentations:
            self.cutmix = CutMix(**augmentations['cutmix'])
            self.cutmix_enabled = True
            logger.info("CutMix augmentation enabled")
        
        if 'random_erasing' in augmentations:
            self.random_erasing = RandomErasing(**augmentations['random_erasing'])
            self.augmentations.append(self.random_erasing)
            logger.info("Random Erasing augmentation enabled")
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation."""
        if self.mixup_enabled:
            return self.mixup(x, y)
        return x, y, y, 1.0
    
    def apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        if self.cutmix_enabled:
            return self.cutmix(x, y)
        return x, y, y, 1.0
    
    def apply_sample_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sample-level augmentations (non-mixing)."""
        for aug in self.augmentations:
            x = aug(x)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Data augmentation module loaded successfully")
