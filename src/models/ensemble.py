"""
Ensemble module for FashionMNIST-Analysis.

Implements various ensemble methods for improved predictions:
- Hard voting
- Soft voting (probability averaging)
- Stacking with meta-learner
- Bagging
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class EnsembleVoting:
    """
    Ensemble voting classifier combining multiple models.
    
    Supports hard voting (majority) and soft voting (average probabilities).
    """
    
    def __init__(self, models: List[nn.Module], voting: str = "soft", 
                 device: str = "cpu", weights: Optional[List[float]] = None):
        """
        Initialize ensemble voting.
        
        Args:
            models (List[nn.Module]): List of models to ensemble
            voting (str): 'hard' for majority voting, 'soft' for probability averaging
            device (str): Device to compute on
            weights (List[float], optional): Weights for each model in soft voting
        """
        self.models = [m.to(device).eval() for m in models]
        self.voting = voting
        self.device = torch.device(device)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(weights) == self.num_models
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        logger.info(f"Ensemble initialized with {self.num_models} models ({voting} voting)")
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions using ensemble.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if self.voting == "hard":
            return self._hard_voting(x)
        else:
            return self._soft_voting(x)
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            np.ndarray: Probability predictions
        """
        probas = []
        
        with torch.no_grad():
            for model, weight in zip(self.models, self.weights):
                logits = model(x)
                proba = torch.softmax(logits, dim=1)
                probas.append(proba.cpu().numpy() * weight)
        
        # Average probabilities
        ensemble_proba = np.mean(probas, axis=0)
        return ensemble_proba
    
    def _hard_voting(self, x: torch.Tensor) -> np.ndarray:
        """Majority voting."""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.append(preds)
        
        predictions = np.array(predictions)
        # Majority vote
        ensemble_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                           axis=0, arr=predictions)
        return ensemble_pred
    
    def _soft_voting(self, x: torch.Tensor) -> np.ndarray:
        """Soft voting using average probabilities."""
        proba = self.predict_proba(x)
        return np.argmax(proba, axis=1)
    
    def get_model_predictions(self, x: torch.Tensor) -> Dict[int, np.ndarray]:
        """
        Get individual model predictions.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Dict[int, np.ndarray]: Predictions from each model
        """
        model_preds = {}
        
        with torch.no_grad():
            for idx, model in enumerate(self.models):
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                model_preds[idx] = preds
        
        return model_preds


class StackingEnsemble:
    """
    Stacking ensemble using a meta-learner.
    
    Base models' predictions are used as features for a meta-learner.
    """
    
    def __init__(self, base_models: List[nn.Module], num_classes: int = 10,
                 device: str = "cpu"):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models (List[nn.Module]): List of base models
            num_classes (int): Number of output classes
            device (str): Device to compute on
        """
        self.base_models = [m.to(device).eval() for m in base_models]
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Meta-learner (using sklearn's LogisticRegression)
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False
        
        logger.info(f"Stacking ensemble initialized with {len(base_models)} base models")
    
    def fit(self, x: torch.Tensor, y: np.ndarray) -> None:
        """
        Fit meta-learner on validation data.
        
        Args:
            x (torch.Tensor): Input tensor
            y (np.ndarray): Target labels
        """
        logger.info("Fitting meta-learner...")
        
        # Generate meta-features from base models
        meta_features = self._generate_meta_features(x)
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, y)
        self.is_fitted = True
        
        logger.info("Meta-learner fitted successfully")
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions using stacking ensemble.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before prediction")
        
        meta_features = self._generate_meta_features(x)
        predictions = self.meta_learner.predict(meta_features)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Meta-learner must be fitted before prediction")
        
        meta_features = self._generate_meta_features(x)
        probas = self.meta_learner.predict_proba(meta_features)
        return probas
    
    def _generate_meta_features(self, x: torch.Tensor) -> np.ndarray:
        """Generate features from base models for meta-learner."""
        meta_features = []
        
        with torch.no_grad():
            for model in self.base_models:
                logits = model(x)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                meta_features.append(proba)
        
        # Concatenate all probabilities
        meta_features = np.hstack(meta_features)
        return meta_features


class BaggingEnsemble:
    """
    Bootstrap Aggregating (Bagging) ensemble.
    
    Trains models on bootstrap samples and averages predictions.
    """
    
    def __init__(self, model_class: type, num_models: int = 5, 
                 num_classes: int = 10, device: str = "cpu"):
        """
        Initialize bagging ensemble.
        
        Args:
            model_class (type): Model class to instantiate
            num_models (int): Number of models to train
            num_classes (int): Number of output classes
            device (str): Device to compute on
        """
        self.model_class = model_class
        self.num_models = num_models
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.models = []
        
        logger.info(f"Bagging ensemble initialized with {num_models} models")
    
    def add_model(self, model: nn.Module) -> None:
        """Add a trained model to ensemble."""
        self.models.append(model.to(self.device).eval())
        logger.info(f"Model {len(self.models)} added to ensemble")
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make predictions using bagging ensemble.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.models:
            raise RuntimeError("No models in ensemble")
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.append(preds)
        
        predictions = np.array(predictions)
        # Average predictions (soft voting via argmax of mean probabilities)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                  axis=0, arr=predictions)
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.models:
            raise RuntimeError("No models in ensemble")
        
        probas = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                probas.append(proba)
        
        # Average probabilities
        ensemble_proba = np.mean(probas, axis=0)
        return ensemble_proba


def evaluate_ensemble(ensemble, test_loader: torch.utils.data.DataLoader,
                     device: str = "cpu") -> Tuple[float, float]:
    """
    Evaluate ensemble on test set.
    
    Args:
        ensemble: Ensemble object
        test_loader: DataLoader for test data
        device (str): Device to compute on
        
    Returns:
        Tuple[float, float]: Accuracy and loss
    """
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = ensemble.predict(x)
            correct += (preds == y.numpy()).sum()
            total += y.size(0)
    
    accuracy = correct / total
    logger.info(f"Ensemble accuracy: {accuracy:.4f}")
    return accuracy, 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Ensemble module loaded successfully")
