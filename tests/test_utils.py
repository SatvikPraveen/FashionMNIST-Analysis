"""
Unit tests for utility functions.
"""

import pytest
import torch
import numpy as np
from src.training.utils import train_step, validation_step, test_step


class TestUtilFunctions:
    """Test utility functions."""
    
    def test_train_step(self):
        """Test training step."""
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        
        # Create dummy data
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Train step
        loss, acc = train_step(model, dataloader, loss_fn, optimizer, device)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1
    
    def test_validation_step(self):
        """Test validation step."""
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        
        # Create dummy data
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Validation step
        loss, acc = validation_step(model, dataloader, loss_fn, device)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1
    
    def test_test_step(self):
        """Test testing step."""
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        loss_fn = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        
        # Create dummy data
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # Test step
        loss, acc = test_step(model, dataloader, loss_fn, device)
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 1


class TestBatchProcessing:
    """Test batch processing utilities."""
    
    def test_batch_tensor_creation(self):
        """Test batch tensor creation."""
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        
        dataset = torch.utils.data.TensorDataset(x, y)
        assert len(dataset) == 16
    
    def test_dataloader_batching(self):
        """Test dataloader batching."""
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        total_samples = 0
        for x_batch, y_batch in dataloader:
            total_samples += x_batch.shape[0]
            assert x_batch.shape[1:] == (1, 28, 28)
            assert y_batch.shape[0] == x_batch.shape[0]
        
        assert total_samples == 100


class TestEvaluation:
    """Test evaluation utilities."""
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        model.eval()
        
        x = torch.randn(16, 1, 28, 28)
        y = torch.randint(0, 10, (16,))
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (16, 10)
        predictions = torch.argmax(output, dim=1)
        assert predictions.shape == (16,)
    
    def test_confusion_matrix_calculation(self):
        """Test confusion matrix calculation."""
        from sklearn.metrics import confusion_matrix
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (3, 3)
        assert cm.sum() == 6
    
    def test_classification_metrics(self):
        """Test classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


class TestModelSaving:
    """Test model saving and loading."""
    
    def test_model_save_load(self, tmp_path):
        """Test saving and loading model."""
        from src.models.architectures import MiniCNN
        
        model = MiniCNN(in_channels=1, num_classes=10)
        model_path = tmp_path / "model.pth"
        
        # Save
        torch.save(model.state_dict(), model_path)
        
        # Load
        model2 = MiniCNN(in_channels=1, num_classes=10)
        model2.load_state_dict(torch.load(model_path))
        
        # Check same
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        
        assert torch.allclose(out1, out2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
