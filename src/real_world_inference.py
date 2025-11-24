"""
Real-world inference module for FashionMNIST-Analysis.

Provides utilities for preprocessing real images and making predictions
on diverse input formats and sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocessing utilities for real-world images.
    
    Handles various image formats, resizing, normalization, and format conversion.
    """
    
    # Fashion MNIST class names
    CLASS_NAMES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot"
    ]
    
    def __init__(self, target_size: int = 224, normalize: bool = True,
                 convert_to_rgb: bool = True, device: str = "cpu"):
        """
        Initialize preprocessor.
        
        Args:
            target_size (int): Target image size (square)
            normalize (bool): Apply ImageNet normalization
            convert_to_rgb (bool): Convert grayscale to RGB
            device (str): Device for preprocessing
        """
        self.target_size = target_size
        self.normalize = normalize
        self.convert_to_rgb = convert_to_rgb
        self.device = torch.device(device)
        
        # ImageNet normalization statistics
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path (str or Path): Path to image file
            
        Returns:
            np.ndarray: Loaded image (H, W, C)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Try PIL first, then OpenCV
        try:
            img = Image.open(image_path)
            img = np.array(img)
        except:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Loaded image: {image_path} (shape: {img.shape})")
        return img
    
    def preprocess(self, image: Union[np.ndarray, str, Path],
                   return_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess image for model inference.
        
        Args:
            image: Image as numpy array, file path, or PIL Image
            return_tensor (bool): Return as tensor if True, else as numpy
            
        Returns:
            Preprocessed image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Handle different image formats
        if len(image.shape) == 2:
            # Grayscale
            image = np.stack([image] * 3, axis=2)
        elif image.shape[2] == 4:
            # RGBA -> RGB
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported image channels: {image.shape[2]}")
        
        # Resize to target size
        image = cv2.resize(image, (self.target_size, self.target_size))
        
        # Normalize to 0-1
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and rearrange (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        # Apply normalization
        if self.normalize:
            image_tensor = (image_tensor - self.mean) / self.std
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        if return_tensor:
            return image_tensor
        else:
            return image_tensor.numpy()
    
    def batch_preprocess(self, images: List[Union[np.ndarray, str, Path]],
                        return_tensor: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess multiple images.
        
        Args:
            images: List of images
            return_tensor (bool): Return as tensor
            
        Returns:
            Batch of preprocessed images
        """
        processed = []
        for img in images:
            proc = self.preprocess(img, return_tensor=True)
            processed.append(proc)
        
        batch = torch.cat(processed, dim=0)
        
        if return_tensor:
            return batch
        else:
            return batch.numpy()


class RealWorldInference:
    """
    Inference on real-world images with confidence scores and predictions.
    """
    
    def __init__(self, model: nn.Module, preprocessor: ImagePreprocessor,
                 device: str = "cpu", confidence_threshold: float = 0.5):
        """
        Initialize inference engine.
        
        Args:
            model (nn.Module): Trained model
            preprocessor (ImagePreprocessor): Image preprocessor
            device (str): Device for inference
            confidence_threshold (float): Minimum confidence for prediction
        """
        self.model = model.to(device).eval()
        self.preprocessor = preprocessor
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
    
    def predict(self, image: Union[np.ndarray, str, Path],
               return_top_k: int = 3) -> dict:
        """
        Make prediction on image.
        
        Args:
            image: Input image
            return_top_k (int): Return top K predictions
            
        Returns:
            dict: Predictions with confidence scores
        """
        # Preprocess
        image_tensor = self.preprocessor.preprocess(image, return_tensor=True)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Get predictions
        probs = probabilities.cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:return_top_k]
        
        predictions = {
            "top_k_predictions": [
                {
                    "class_id": int(idx),
                    "class_name": ImagePreprocessor.CLASS_NAMES[idx],
                    "confidence": float(probs[idx])
                }
                for idx in top_indices
            ],
            "predicted_class": int(top_indices[0]),
            "predicted_class_name": ImagePreprocessor.CLASS_NAMES[top_indices[0]],
            "confidence": float(probs[top_indices[0]]),
            "all_probabilities": {
                ImagePreprocessor.CLASS_NAMES[i]: float(probs[i])
                for i in range(len(probs))
            }
        }
        
        return predictions
    
    def predict_batch(self, images: List[Union[np.ndarray, str, Path]],
                     return_top_k: int = 1) -> List[dict]:
        """
        Make predictions on batch of images.
        
        Args:
            images: List of images
            return_top_k (int): Return top K predictions
            
        Returns:
            List[dict]: Predictions for each image
        """
        batch = self.preprocessor.batch_preprocess(images, return_tensor=True)
        batch = batch.to(self.device)
        
        with torch.no_grad():
            logits = self.model(batch)
            probabilities = F.softmax(logits, dim=1)
        
        probs = probabilities.cpu().numpy()
        predictions = []
        
        for i in range(len(images)):
            top_indices = np.argsort(probs[i])[::-1][:return_top_k]
            pred = {
                "top_k_predictions": [
                    {
                        "class_id": int(idx),
                        "class_name": ImagePreprocessor.CLASS_NAMES[idx],
                        "confidence": float(probs[i, idx])
                    }
                    for idx in top_indices
                ],
                "predicted_class": int(top_indices[0]),
                "confidence": float(probs[i, top_indices[0]])
            }
            predictions.append(pred)
        
        return predictions
    
    def predict_with_uncertainty(self, image: Union[np.ndarray, str, Path],
                                num_samples: int = 10) -> dict:
        """
        Make prediction with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            image: Input image
            num_samples (int): Number of stochastic forward passes
            
        Returns:
            dict: Prediction with uncertainty estimates
        """
        image_tensor = self.preprocessor.preprocess(image, return_tensor=True)
        image_tensor = image_tensor.to(self.device)
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.model(image_tensor)
                probs = F.softmax(logits, dim=1)
                predictions_list.append(probs.cpu().numpy()[0])
        
        self.model.eval()
        
        predictions_array = np.array(predictions_list)
        mean_probs = predictions_array.mean(axis=0)
        std_probs = predictions_array.std(axis=0)
        
        pred_class = np.argmax(mean_probs)
        
        result = {
            "predicted_class": int(pred_class),
            "predicted_class_name": ImagePreprocessor.CLASS_NAMES[pred_class],
            "mean_confidence": float(mean_probs[pred_class]),
            "confidence_std": float(std_probs[pred_class]),
            "uncertainty": float(np.max(std_probs)),
            "mean_probabilities": mean_probs.tolist(),
            "std_probabilities": std_probs.tolist()
        }
        
        return result


def load_and_preprocess_dataset(image_dir: Union[str, Path],
                               preprocessor: ImagePreprocessor) -> Tuple[torch.Tensor, List[str]]:
    """
    Load all images from directory and preprocess.
    
    Args:
        image_dir: Directory containing images
        preprocessor: Image preprocessor
        
    Returns:
        Tuple of preprocessed images and file paths
    """
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))
    
    images = []
    paths = []
    
    for img_path in image_files:
        try:
            img_tensor = preprocessor.preprocess(img_path, return_tensor=True)
            images.append(img_tensor)
            paths.append(str(img_path))
        except Exception as e:
            logger.warning(f"Failed to process {img_path}: {e}")
    
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    
    batch = torch.cat(images, dim=0)
    logger.info(f"Loaded and preprocessed {len(images)} images")
    
    return batch, paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocessor = ImagePreprocessor(target_size=224)
    logger.info("Real-world inference module loaded successfully")
