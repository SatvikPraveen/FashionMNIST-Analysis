"""
Monitoring module for FashionMNIST-Analysis.

Provides utilities for tracking metrics, detecting drift, and monitoring model performance.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track metrics during training and inference.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size (int): Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics = {}
        self.history = {}
    
    def update(self, metric_name: str, value: float) -> None:
        """
        Update a metric.
        
        Args:
            metric_name (str): Name of metric
            value (float): Metric value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
            self.history[metric_name] = []
        
        self.metrics[metric_name].append(value)
        self.history[metric_name].append(value)
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """
        Get current metric value (average of window).
        
        Args:
            metric_name (str): Name of metric
            
        Returns:
            Average metric value
        """
        if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
            return np.mean(list(self.metrics[metric_name]))
        return None
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all current metrics."""
        return {
            name: np.mean(list(values))
            for name, values in self.metrics.items()
            if len(values) > 0
        }
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get full history of metric."""
        return self.history.get(metric_name, [])


class DriftDetector:
    """
    Detect distribution shift (data drift) in model inputs or outputs.
    """
    
    def __init__(self, reference_data: Optional[np.ndarray] = None,
                 threshold: float = 0.5, method: str = "kl_divergence"):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference distribution (for comparison)
            threshold: Drift detection threshold
            method: Drift detection method ('kl_divergence', 'wasserstein', 'ks_test')
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.method = method
        self.recent_data = []
        
        if reference_data is not None:
            self.reference_mean = reference_data.mean(axis=0)
            self.reference_std = reference_data.std(axis=0)
        else:
            self.reference_mean = None
            self.reference_std = None
    
    def update_reference(self, data: np.ndarray) -> None:
        """
        Update reference distribution.
        
        Args:
            data: New reference data
        """
        self.reference_data = data
        self.reference_mean = data.mean(axis=0)
        self.reference_std = data.std(axis=0)
        logger.info("Reference distribution updated")
    
    def detect_drift(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Detect drift in data.
        
        Args:
            data: New data to check
            
        Returns:
            Tuple of (drift_detected, drift_score)
        """
        if self.reference_data is None:
            logger.warning("Reference data not set, skipping drift detection")
            return False, 0.0
        
        if self.method == "kl_divergence":
            score = self._kl_divergence(data)
        elif self.method == "wasserstein":
            score = self._wasserstein_distance(data)
        elif self.method == "ks_test":
            score = self._ks_test(data)
        else:
            raise ValueError(f"Unknown drift method: {self.method}")
        
        drift_detected = score > self.threshold
        
        if drift_detected:
            logger.warning(f"Drift detected! Score: {score:.4f} (threshold: {self.threshold})")
        
        return drift_detected, score
    
    def _kl_divergence(self, data: np.ndarray) -> float:
        """Compute KL divergence between distributions."""
        from scipy.stats import entropy
        
        # Create histograms
        p, _ = np.histogram(self.reference_data.flatten(), bins=20)
        q, _ = np.histogram(data.flatten(), bins=20)
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        
        return entropy(p, q)
    
    def _wasserstein_distance(self, data: np.ndarray) -> float:
        """Compute Wasserstein distance between distributions."""
        from scipy.stats import wasserstein_distance
        
        return wasserstein_distance(
            self.reference_data.flatten(),
            data.flatten()
        )
    
    def _ks_test(self, data: np.ndarray) -> float:
        """Perform Kolmogorov-Smirnov test."""
        from scipy.stats import ks_2samp
        
        stat, _ = ks_2samp(
            self.reference_data.flatten(),
            data.flatten()
        )
        return stat


class PredictionMonitor:
    """
    Monitor model predictions for anomalies and performance degradation.
    """
    
    def __init__(self, num_classes: int = 10, window_size: int = 100):
        """
        Initialize prediction monitor.
        
        Args:
            num_classes (int): Number of output classes
            window_size (int): Size of rolling window
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.ground_truth = deque(maxlen=window_size)
    
    def update(self, pred: np.ndarray, confidence: np.ndarray, 
               ground_truth: Optional[np.ndarray] = None) -> None:
        """
        Update monitor with new predictions.
        
        Args:
            pred: Predicted class
            confidence: Prediction confidence
            ground_truth: True label (optional)
        """
        self.predictions.append(pred)
        self.confidences.append(confidence)
        
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get prediction statistics.
        
        Returns:
            Dictionary of statistics
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences = list(self.confidences)
        
        stats = {
            "mean_confidence": np.mean(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "predictions_seen": len(self.predictions)
        }
        
        if len(self.ground_truth) == len(self.predictions):
            acc = np.mean([p == gt for p, gt in zip(self.predictions, self.ground_truth)])
            stats["accuracy"] = acc
        
        return stats
    
    def detect_low_confidence(self, threshold: float = 0.5) -> List[int]:
        """
        Find predictions with low confidence.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Indices of low confidence predictions
        """
        return [
            i for i, conf in enumerate(self.confidences)
            if conf < threshold
        ]


class PerformanceLogger:
    """
    Log performance metrics to file.
    """
    
    def __init__(self, log_dir: str = "./logs"):
        """
        Initialize logger.
        
        Args:
            log_dir (str): Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = None
    
    def start_session(self, session_name: str = "training") -> str:
        """
        Start a new logging session.
        
        Args:
            session_name (str): Name of session
            
        Returns:
            Session ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{session_name}_{timestamp}"
        self.current_log = {
            "session_id": session_id,
            "start_time": timestamp,
            "metrics": []
        }
        logger.info(f"Started logging session: {session_id}")
        return session_id
    
    def log_metric(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for an epoch.
        
        Args:
            epoch (int): Epoch number
            metrics (Dict): Dictionary of metrics
        """
        if self.current_log is None:
            logger.warning("No active logging session")
            return
        
        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.current_log["metrics"].append(entry)
    
    def save_session(self) -> None:
        """Save current session to file."""
        if self.current_log is None:
            return
        
        filename = self.log_dir / f"{self.current_log['session_id']}.json"
        with open(filename, 'w') as f:
            json.dump(self.current_log, f, indent=2)
        
        logger.info(f"Session saved to {filename}")
    
    def get_session_summary(self) -> Dict:
        """Get summary of current session."""
        if self.current_log is None:
            return {}
        
        metrics = self.current_log["metrics"]
        if not metrics:
            return {}
        
        # Compute statistics
        keys = set()
        for m in metrics:
            keys.update(m.keys())
        keys.discard("epoch")
        keys.discard("timestamp")
        
        summary = {}
        for key in keys:
            values = [m[key] for m in metrics if key in m]
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
        
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Monitoring module loaded successfully")
