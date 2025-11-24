"""
Unit tests for model definitions and utilities.
"""

import pytest
import torch
import numpy as np
from src.model_definitions import ResNet, BasicBlock, MiniCNN, TinyVGG
from src.transfer_learning import TransferLearningModel, load_vit_model
from src.ensemble import EnsembleVoting, StackingEnsemble
from src.data_augmentation import Mixup, CutMix, RandomErasing
from src.monitoring import MetricsTracker, DriftDetector


class TestModelDefinitions:
    """Test model architectures."""
    
    def test_resnet_output_shape(self):
        """Test ResNet output shape."""
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        x = torch.randn(4, 1, 28, 28)  # Batch of 4 grayscale images
        output = model(x)
        assert output.shape == (4, 10)
    
    def test_minicnn_output_shape(self):
        """Test MiniCNN output shape."""
        model = MiniCNN(in_channels=1, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)
    
    def test_tinyvgg_output_shape(self):
        """Test TinyVGG output shape."""
        model = TinyVGG(in_channels=1, hidden_units=32, num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)
    
    def test_model_device_transfer(self):
        """Test model can be moved to different devices."""
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        # Try CPU
        model.to("cpu")
        x = torch.randn(2, 1, 28, 28)
        output = model(x)
        assert output.shape == (2, 10)


class TestTransferLearning:
    """Test transfer learning utilities."""
    
    def test_transfer_learning_model_init(self):
        """Test TransferLearningModel initialization."""
        # Note: This will fail if timm is not installed
        try:
            tl = TransferLearningModel("resnet50", num_classes=10, pretrained=False)
            assert tl.model is not None
            assert tl.device is not None
        except ImportError:
            pytest.skip("TIMM not installed")
    
    def test_freeze_backbone(self):
        """Test freezing backbone parameters."""
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        # Initially all params should be trainable
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Freeze all but last layer
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[0].requires_grad = True
        
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert trainable_after < trainable_before


class TestEnsemble:
    """Test ensemble methods."""
    
    def test_ensemble_voting_hard(self):
        """Test hard voting ensemble."""
        model1 = MiniCNN(1, 10)
        model2 = MiniCNN(1, 10)
        
        ensemble = EnsembleVoting([model1, model2], voting="hard")
        
        x = torch.randn(4, 1, 28, 28)
        predictions = ensemble.predict(x)
        
        assert predictions.shape == (4,)
        assert np.all((predictions >= 0) & (predictions < 10))
    
    def test_ensemble_voting_soft(self):
        """Test soft voting ensemble."""
        model1 = MiniCNN(1, 10)
        model2 = MiniCNN(1, 10)
        
        ensemble = EnsembleVoting([model1, model2], voting="soft")
        
        x = torch.randn(4, 1, 28, 28)
        predictions = ensemble.predict(x)
        
        assert predictions.shape == (4,)
        assert np.all((predictions >= 0) & (predictions < 10))
    
    def test_ensemble_probabilities(self):
        """Test ensemble probability predictions."""
        model1 = MiniCNN(1, 10)
        model2 = MiniCNN(1, 10)
        
        ensemble = EnsembleVoting([model1, model2], voting="soft")
        
        x = torch.randn(2, 1, 28, 28)
        proba = ensemble.predict_proba(x)
        
        assert proba.shape == (2, 10)
        # Probabilities should sum to 1
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestDataAugmentation:
    """Test data augmentation."""
    
    def test_mixup(self):
        """Test Mixup augmentation."""
        mixup = Mixup(alpha=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3])
        
        x_mixed, y_a, y_b, lam = mixup(x, y)
        
        assert x_mixed.shape == x.shape
        assert lam >= 0 and lam <= 1
    
    def test_cutmix(self):
        """Test CutMix augmentation."""
        cutmix = CutMix(alpha=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 3])
        
        x_mixed, y_a, y_b, lam = cutmix(x, y)
        
        assert x_mixed.shape == x.shape
        assert lam >= 0 and lam <= 1
    
    def test_random_erasing(self):
        """Test Random Erasing augmentation."""
        eraser = RandomErasing(probability=1.0)
        
        x = torch.randn(4, 3, 224, 224)
        x_aug = eraser(x)
        
        assert x_aug.shape == x.shape
        # Should have erased pixels
        assert not torch.allclose(x, x_aug)


class TestMonitoring:
    """Test monitoring utilities."""
    
    def test_metrics_tracker(self):
        """Test metrics tracker."""
        tracker = MetricsTracker(window_size=10)
        
        # Add metrics
        for i in range(5):
            tracker.update("loss", 0.1 + i * 0.01)
            tracker.update("accuracy", 0.8 + i * 0.02)
        
        # Check we can retrieve metrics
        loss = tracker.get_metric("loss")
        acc = tracker.get_metric("accuracy")
        
        assert loss is not None
        assert acc is not None
        assert acc > loss  # Accuracy should be higher than loss
    
    def test_drift_detector(self):
        """Test drift detector."""
        # Reference data (normal distribution)
        ref_data = np.random.normal(0, 1, (100, 10))
        
        detector = DriftDetector(reference_data=ref_data, threshold=0.5)
        
        # Test data from same distribution
        test_data = np.random.normal(0, 1, (50, 10))
        drift_detected, score = detector.detect_drift(test_data)
        
        # Score should be low (no drift)
        assert score < detector.threshold
        assert not drift_detected
    
    def test_prediction_monitor(self):
        """Test prediction monitor."""
        from src.monitoring import PredictionMonitor
        
        monitor = PredictionMonitor(num_classes=10, window_size=50)
        
        # Simulate predictions
        for i in range(20):
            pred = np.random.randint(0, 10)
            confidence = np.random.uniform(0.5, 1.0)
            true_label = np.random.randint(0, 10)
            
            monitor.update(pred, confidence, true_label)
        
        # Get statistics
        stats = monitor.get_statistics()
        
        assert "mean_confidence" in stats
        assert "accuracy" in stats
        assert stats["mean_confidence"] >= 0.5  # Min confidence was 0.5


class TestConfig:
    """Test configuration system."""
    
    def test_config_loading(self):
        """Test config loading."""
        from src.config import load_config
        
        config = load_config("config.yaml")
        
        assert config.model.num_classes == 10
        assert config.training.batch_size > 0
    
    def test_config_get_with_dot_notation(self):
        """Test config get with dot notation."""
        from src.config import load_config
        
        config = load_config("config.yaml")
        
        batch_size = config.get("training.batch_size")
        assert batch_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
