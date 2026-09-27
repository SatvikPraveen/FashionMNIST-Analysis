"""
Tests for seeding and for sample-weighted metric accumulation.
"""

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.architectures import MiniCNN
from src.training.reproducibility import set_seed, make_generator
from src.training.utils import train_step, validation_step, test_step as run_test_step
from src.training.trainer import train_epoch_with_augmentation
from src.evaluation.metrics import evaluate_model_with_confusion_matrix


def _uneven_loader(n=50, batch_size=32, seed=0):
    """50 samples / batch 32 -> a full batch of 32 and a partial batch of 18."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False), x, y


def _reference_accuracy(model, x, y):
    model.eval()
    with torch.inference_mode():
        preds = model(x).argmax(dim=1)
    return accuracy_score(y.numpy(), preds.numpy())


class TestSampleWeightedAccuracy:
    """The last, partial batch must count by its size, not as a full batch."""

    def test_validation_step_matches_sklearn(self):
        set_seed(0)
        model = MiniCNN(in_channels=1, num_classes=10)
        loader, x, y = _uneven_loader()
        _, acc = validation_step(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        assert acc == pytest.approx(_reference_accuracy(model, x, y), abs=1e-9)

    def test_test_step_matches_sklearn(self):
        set_seed(0)
        model = MiniCNN(in_channels=1, num_classes=10)
        loader, x, y = _uneven_loader()
        _, acc = run_test_step(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        assert acc == pytest.approx(_reference_accuracy(model, x, y), abs=1e-9)

    def test_evaluate_with_confusion_matrix_matches_sklearn(self):
        set_seed(0)
        model = MiniCNN(in_channels=1, num_classes=10)
        loader, x, y = _uneven_loader()
        _, acc, preds, labels = evaluate_model_with_confusion_matrix(model, loader, torch.device("cpu"))
        assert acc == pytest.approx(_reference_accuracy(model, x, y), abs=1e-9)
        assert len(preds) == len(labels) == 50

    def test_per_batch_mean_would_be_wrong(self):
        """Regression guard: build a case where batch-mean != sample-mean."""
        # Batch 1: 4 samples, all correct. Batch 2: 1 sample, wrong.
        # Sample-weighted accuracy = 4/5 = 0.8; per-batch mean = (1 + 0)/2 = 0.5.
        class Fixed(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros(x.shape[0], 10)
                logits[:, 0] = 1.0  # always predicts class 0
                return logits

        x = torch.zeros(5, 1, 28, 28)
        y = torch.tensor([0, 0, 0, 0, 1])
        loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)
        _, acc = run_test_step(Fixed(), loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        assert acc == pytest.approx(0.8)

    def test_train_steps_report_sample_weighted_accuracy(self):
        set_seed(0)
        model = MiniCNN(in_channels=1, num_classes=10)
        loader, _, _ = _uneven_loader()
        loss_fn = torch.nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0: weights unchanged
        _, acc_plain = train_step(model, loader, loss_fn, opt, torch.device("cpu"))
        _, acc_aug = train_epoch_with_augmentation(model, loader, loss_fn, opt, torch.device("cpu"))
        assert 0.0 <= acc_plain <= 1.0 and 0.0 <= acc_aug <= 1.0
        # Both should be k/50 for some integer k, never a per-batch average
        assert (acc_plain * 50) == pytest.approx(round(acc_plain * 50), abs=1e-9)
        assert (acc_aug * 50) == pytest.approx(round(acc_aug * 50), abs=1e-9)


class TestSeeding:
    def test_set_seed_returns_seed(self):
        assert set_seed(123) == 123

    def test_same_seed_same_init(self):
        set_seed(7)
        m1 = MiniCNN(in_channels=1, num_classes=10)
        set_seed(7)
        m2 = MiniCNN(in_channels=1, num_classes=10)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seed_different_init(self):
        set_seed(7)
        m1 = MiniCNN(in_channels=1, num_classes=10)
        set_seed(8)
        m2 = MiniCNN(in_channels=1, num_classes=10)
        assert any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))

    def test_same_seed_same_shuffle_order(self):
        ds = TensorDataset(torch.arange(100).float().view(100, 1), torch.zeros(100, dtype=torch.long))
        order = []
        for _ in range(2):
            loader = DataLoader(ds, batch_size=100, shuffle=True, generator=make_generator(3))
            order.append(next(iter(loader))[0].squeeze().tolist())
        assert order[0] == order[1]

    def test_numpy_and_python_random_are_seeded(self):
        import random
        set_seed(11); a = (np.random.rand(), random.random())
        set_seed(11); b = (np.random.rand(), random.random())
        assert a == b

    def test_make_generator_none(self):
        assert make_generator(None) is None
