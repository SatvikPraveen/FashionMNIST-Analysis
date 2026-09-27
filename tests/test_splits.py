"""
One train/validation split everywhere: the trainer, the prepared CSVs and
evaluation must agree, or temperature scaling leaks training data.
"""

import json

import pytest
import torch
from torch.utils.data import TensorDataset

from src.training.reproducibility import split_indices, make_generator
from src.data.preparation import split_train_val
from src.evaluation import evaluate as ev


def test_split_indices_reproduces_random_split():
    """Runs made before split_indices existed must keep their exact split."""
    n = 1000
    ds = TensorDataset(torch.arange(n))
    tr, va = torch.utils.data.random_split(ds, [800, 200], generator=make_generator(42))
    new_tr, new_va = split_indices(n, 0.2, 42)
    assert list(tr.indices) == new_tr and list(va.indices) == new_va


def test_split_is_disjoint_and_complete():
    tr, va = split_indices(500, 0.2, 7)
    assert len(tr) == 400 and len(va) == 100
    assert set(tr).isdisjoint(va) and set(tr) | set(va) == set(range(500))


def test_different_seeds_give_different_splits():
    assert split_indices(500, 0.2, 0)[1] != split_indices(500, 0.2, 1)[1]


class _FakeImages:
    """Indexable (image, label) dataset where each image encodes its index."""
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        import numpy as np
        return np.full((28, 28), i % 256, dtype=np.uint8), i % 10


def test_prepared_csv_split_matches_training_split():
    (tr_img, tr_lab), (va_img, va_lab) = split_train_val(_FakeImages(300), 0.8, seed=5)
    _, va_idx = split_indices(300, 0.2, 5)
    assert [int(img[0]) for img in va_img] == [i % 256 for i in va_idx]
    assert va_lab == [i % 10 for i in va_idx]


def _write(path, obj):
    path.write_text(json.dumps(obj))


def test_evaluator_uses_csv_only_when_seeds_match(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"; run_dir.mkdir()
    data_dir = tmp_path / "data"; data_dir.mkdir()
    ckpt = run_dir / "tinyvgg_best.pth"; ckpt.write_bytes(b"")
    val_csv = data_dir / "fashion_mnist_val.csv"
    val_csv.write_text("label," + ",".join(str(i) for i in range(784)) + "\n" + "3," + ",".join("0" * 784) + "\n")
    _write(data_dir / "split.json", {"seed": 42})

    rebuilt = []
    monkeypatch.setattr("src.data.dataset.create_dataloaders",
                        lambda **kw: rebuilt.append(kw["seed"]) or (None, "OWN_SPLIT", None))

    _write(run_dir / "run.json", {"params": {"seed": 42}})
    loader = ev.resolve_val_loader(str(ckpt), str(val_csv), 16)
    assert loader != "OWN_SPLIT" and len(loader.dataset) == 1 and rebuilt == []

    _write(run_dir / "run.json", {"params": {"seed": 3}})
    assert ev.resolve_val_loader(str(ckpt), str(val_csv), 16) == "OWN_SPLIT" and rebuilt == [3]


def test_evaluator_warns_when_seed_unknown(tmp_path, capsys):
    ckpt = tmp_path / "m.pth"; ckpt.write_bytes(b"")
    val_csv = tmp_path / "v.csv"
    val_csv.write_text("label," + ",".join(str(i) for i in range(784)) + "\n" + "1," + ",".join("0" * 784) + "\n")
    ev.resolve_val_loader(str(ckpt), str(val_csv), 8)
    assert "may overlap" in capsys.readouterr().out
    assert ev.resolve_val_loader(str(ckpt), None, 8) is None
