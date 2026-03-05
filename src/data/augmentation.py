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


class GaussianNoise:
    """
    Additive Gaussian noise augmentation.

    Adds zero-mean Gaussian noise to a tensor image, simulating sensor noise
    and improving robustness to small perturbations.
    """

    def __init__(self, std: float = 0.05, probability: float = 0.5):
        """
        Initialize Gaussian Noise.

        Args:
            std (float): Standard deviation of the noise (relative to [0,1] pixel range)
            probability (float): Probability of applying noise
        """
        self.std = std
        self.probability = probability

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise.

        Args:
            x (torch.Tensor): Input tensor (C, H, W) or (B, C, H, W)

        Returns:
            Noisy tensor clipped to [0, 1]
        """
        if np.random.rand() > self.probability:
            return x
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)


class GaussianBlur:
    """
    Gaussian blur augmentation using torch.nn.functional.
    """
    
    def __init__(self, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0), probability: float = 0.5):
        """
        Initialize Gaussian Blur.
        
        Args:
            kernel_size (int): Kernel size (must be odd)
            sigma (tuple): Range of sigma values (min, max)
            probability (float): Probability of applying blur
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
        self.probability = probability
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur."""
        if np.random.rand() > self.probability:
            return x
        
        # Use torchvision's GaussianBlur if available
        try:
            from torchvision.transforms import functional as F
            sigma = np.random.uniform(*self.sigma)
            return F.gaussian_blur(x, self.kernel_size, [sigma, sigma])
        except ImportError:
            # Fallback: return unchanged
            logger.warning("torchvision not available, skipping GaussianBlur")
            return x


class TorchvisionTransforms:
    """
    Wrapper for torchvision transforms to use in data pipeline.
    
    Provides common augmentations: rotation, flipping, color jitter, etc.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize torchvision transforms.
        
        Args:
            config (dict): Transform configuration
                Example:
                {
                    'rotation': 15,
                    'horizontal_flip': True,
                    'vertical_flip': False,
                    'color_jitter': {'brightness': 0.2, 'contrast': 0.2},
                    'random_crop': {'size': 28, 'padding': 4}
                }
        """
        from torchvision import transforms
        
        if config is None:
            config = {}
        
        transform_list = []
        
        # Random rotation
        if 'rotation' in config and config['rotation'] > 0:
            transform_list.append(
                transforms.RandomRotation(degrees=config['rotation'])
            )
            logger.info(f"Added RandomRotation: {config['rotation']} degrees")
        
        # Horizontal flip
        if config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            logger.info("Added RandomHorizontalFlip")
        
        # Vertical flip
        if config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
            logger.info("Added RandomVerticalFlip")
        
        # Color jitter
        if 'color_jitter' in config:
            cj = config['color_jitter']
            transform_list.append(
                transforms.ColorJitter(
                    brightness=cj.get('brightness', 0),
                    contrast=cj.get('contrast', 0),
                    saturation=cj.get('saturation', 0),
                    hue=cj.get('hue', 0)
                )
            )
            logger.info("Added ColorJitter")
        
        # Random crop
        if 'random_crop' in config:
            rc = config['random_crop']
            transform_list.append(
                transforms.RandomCrop(
                    size=rc['size'],
                    padding=rc.get('padding', 0)
                )
            )
            logger.info(f"Added RandomCrop: {rc['size']}x{rc['size']}")
        
        # Random affine
        if 'random_affine' in config:
            ra = config['random_affine']
            transform_list.append(
                transforms.RandomAffine(
                    degrees=ra.get('degrees', 0),
                    translate=ra.get('translate', None),
                    scale=ra.get('scale', None)
                )
            )
            logger.info("Added RandomAffine")
        
        self.transform = transforms.Compose(transform_list) if transform_list else None
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transforms."""
        if self.transform is not None:
            return self.transform(x)
        return x


class AlbumentationsTransforms:
    """
    Wrapper for albumentations library (if available).
    
    Provides advanced augmentations specifically for image data.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize albumentations transforms.
        
        Args:
            config (dict): Albumentations configuration
        """
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            if config is None:
                config = {}
            
            transform_list = []
            
            # Add common augmentations based on config
            if config.get('horizontal_flip', False):
                transform_list.append(A.HorizontalFlip(p=0.5))
            
            if config.get('vertical_flip', False):
                transform_list.append(A.VerticalFlip(p=0.5))
            
            if 'rotation' in config:
                transform_list.append(
                    A.Rotate(limit=config['rotation'], p=0.5)
                )
            
            if config.get('shift_scale_rotate', False):
                transform_list.append(
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=15,
                        p=0.5
                    )
                )
            
            self.transform = A.Compose(transform_list) if transform_list else None
            self.available = True
            logger.info("Albumentations transforms initialized")
            
        except ImportError:
            logger.warning("albumentations not available, skipping")
            self.transform = None
            self.available = False
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply albumentations.
        
        Args:
            x (np.ndarray): Image as numpy array
            
        Returns:
            Augmented image
        """
        if self.transform is not None and self.available:
            augmented = self.transform(image=x)
            return augmented['image']
        return x


class AugmentationPipeline:
    """
    Combines multiple augmentations into a comprehensive pipeline.
    
    Supports:
    - Basic transforms (torchvision)
    - Advanced augmentations (Mixup, CutMix, RandomErasing)
    - Custom augmentations (albumentations)
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
                    'random_erasing': {'probability': 0.5},
                    'torchvision': {'rotation': 15, 'horizontal_flip': True},
                    'gaussian_blur': {'kernel_size': 3, 'sigma': (0.1, 2.0)}
                }
        """
        self.augmentations = []
        self.cutmix_enabled = False
        self.mixup_enabled = False
        self.torchvision_aug = None
        
        if augmentations is None:
            augmentations = {}
        
        # Batch-level augmentations (applied to batches during training)
        if 'mixup' in augmentations:
            self.mixup = Mixup(**augmentations['mixup'])
            self.mixup_enabled = True
            logger.info("Mixup augmentation enabled")
        
        if 'cutmix' in augmentations:
            self.cutmix = CutMix(**augmentations['cutmix'])
            self.cutmix_enabled = True
            logger.info("CutMix augmentation enabled")
        
        # Sample-level augmentations (applied per image)
        if 'random_erasing' in augmentations:
            self.random_erasing = RandomErasing(**augmentations['random_erasing'])
            self.augmentations.append(self.random_erasing)
            logger.info("Random Erasing augmentation enabled")
        
        if 'gaussian_blur' in augmentations:
            self.gaussian_blur = GaussianBlur(**augmentations['gaussian_blur'])
            self.augmentations.append(self.gaussian_blur)
            logger.info("Gaussian Blur augmentation enabled")

        if 'gaussian_noise' in augmentations:
            self.gaussian_noise = GaussianNoise(**augmentations['gaussian_noise'])
            self.augmentations.append(self.gaussian_noise)
            logger.info("Gaussian Noise augmentation enabled")
        
        # Torchvision transforms
        if 'torchvision' in augmentations:
            self.torchvision_aug = TorchvisionTransforms(augmentations['torchvision'])
            logger.info("Torchvision transforms enabled")
        
        # Albumentations (if available)
        if 'albumentations' in augmentations:
            self.albumentations_aug = AlbumentationsTransforms(augmentations['albumentations'])
            if self.albumentations_aug.available:
                logger.info("Albumentations enabled")
    
    def apply_mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply Mixup augmentation to batch."""
        if self.mixup_enabled:
            return self.mixup(x, y)
        return x, y, y, 1.0
    
    def apply_cutmix(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation to batch."""
        if self.cutmix_enabled:
            return self.cutmix(x, y)
        return x, y, y, 1.0
    
    def apply_sample_augmentations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sample-level augmentations (non-mixing)."""
        # Apply torchvision transforms first
        if self.torchvision_aug is not None:
            x = self.torchvision_aug(x)
        
        # Apply other augmentations
        for aug in self.augmentations:
            x = aug(x)
        
        return x
    
    def get_batch_augmentation_probability(self) -> float:
        """
        Get probability of applying batch augmentation.
        
        Returns 0.5 if either Mixup or CutMix is enabled.
        """
        if self.mixup_enabled or self.cutmix_enabled:
            return 0.5
        return 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Data augmentation module loaded successfully")
