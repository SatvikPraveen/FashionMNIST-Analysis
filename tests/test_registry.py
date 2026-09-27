"""
Tests for the model registry and for the trainer's use of it.
"""

import json
import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import Config
from src.models.registry import (
    ModelSpec, TimmClassifier, build_model, build_from_spec, resolve_name,
    default_image_size, load_model_from_checkpoint, spec_path_for,
    model_spec_from_config, CUSTOM_MODELS,
)
from src.training.trainer import get_model, train_model


X = torch.randn(2, 1, 28, 28)


class TestResolve:
    def test_custom_names(self):
        for n in CUSTOM_MODELS:
            assert resolve_name(n) == n
            assert resolve_name(n.upper()) == n

    def test_aliases(self):
        assert resolve_name("vit_tiny") == "vit_tiny_patch16_224"
        assert resolve_name("efficientnet") == "efficientnet_b0"

    def test_timm_prefix_passthrough(self):
        assert resolve_name("timm:resnet26d") == "resnet26d"

    def test_default_sizes(self):
        assert default_image_size("tinyvgg") == 28
        assert default_image_size("resnet18") == 64
        assert default_image_size("vit_tiny_patch16_224") == 224
        assert default_image_size("deit_small_patch16_224") == 224


class TestBuild:
    @pytest.mark.parametrize("name", CUSTOM_MODELS)
    def test_custom_models_forward(self, name):
        m = build_model(name)
        assert m(X).shape == (2, 10)

    @pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
    def test_timm_cnn_forward(self, name):
        pytest.importorskip("timm")
        m = build_model(name, pretrained=False)
        assert isinstance(m, TimmClassifier)
        assert m.image_size == 64
        assert m(X).shape == (2, 10)

    def test_timm_vit_forward_with_custom_size(self):
        pytest.importorskip("timm")
        m = build_model("vit_tiny", pretrained=False, image_size=32)
        assert m.image_size == 32
        assert m(X).shape == (2, 10)

    def test_freeze_backbone_leaves_head_trainable(self):
        pytest.importorskip("timm")
        m = build_model("resnet18", pretrained=False, freeze_backbone=True)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert trainable and all("fc" in n for n in trainable)
        m.unfreeze()
        assert all(p.requires_grad for p in m.parameters())

    def test_unknown_name_raises(self):
        pytest.importorskip("timm")
        with pytest.raises(Exception):
            build_model("definitely_not_a_model_xyz")


class TestSpec:
    def test_roundtrip(self, tmp_path):
        spec = ModelSpec(name="resnet18", pretrained=True, image_size=96, drop_rate=0.1)
        path = tmp_path / "spec.json"
        spec.save(str(path))
        loaded = ModelSpec.load(str(path))
        assert loaded == spec
        assert loaded.is_timm

    def test_from_dict_ignores_unknown_keys(self):
        spec = ModelSpec.from_dict({"name": "tinyvgg", "bogus": 1})
        assert spec.name == "tinyvgg" and not spec.is_timm

    def test_spec_path_for(self):
        assert spec_path_for("models/x/x_best.pth") == "models/x/x_best_spec.json"

    def test_checkpoint_roundtrip(self, tmp_path):
        pytest.importorskip("timm")
        m = build_model("resnet18", pretrained=False, image_size=32)
        weights = tmp_path / "m.pth"
        torch.save(m.state_dict(), weights)
        ModelSpec(name="resnet18", image_size=32).save(spec_path_for(str(weights)))
        m2 = load_model_from_checkpoint(str(weights))
        with torch.inference_mode():
            assert torch.allclose(m.eval()(X), m2(X), atol=1e-5)


def _config(tmp_path, **model_overrides):
    """Minimal config file that train_model / get_model can consume."""
    cfg = {
        "model": {"architecture": "tinyvgg", "num_classes": 10, "pretrained": False,
                  "image_size": None, "drop_rate": 0.0, **model_overrides},
        "transfer_learning": {"freeze_backbone": False},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 1e-3,
                     "weight_decay": 0.0, "optimizer": "adam", "scheduler": "none",
                     "early_stopping_patience": 5, "seed": 0},
        "augmentation": {"enabled": False},
    }
    import yaml
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return Config(str(path))


class TestTrainerIntegration:
    def test_get_model_uses_config_flags(self, tmp_path):
        pytest.importorskip("timm")
        cfg = _config(tmp_path, pretrained=False, image_size=32, drop_rate=0.2)
        m = get_model("resnet18", config=cfg)
        assert isinstance(m, TimmClassifier) and m.image_size == 32
        assert m.spec.drop_rate == 0.2 and m.spec.pretrained is False

    def test_model_spec_from_config_custom_ignores_pretrained(self, tmp_path):
        cfg = _config(tmp_path, pretrained=True)
        spec = model_spec_from_config("tinyvgg", cfg)
        assert spec.pretrained is False and spec.image_size is None

    @pytest.mark.parametrize("name", ["tinyvgg", "resnet18"])
    def test_train_model_writes_spec_and_history(self, tmp_path, name):
        if name != "tinyvgg":
            pytest.importorskip("timm")
        cfg = _config(tmp_path, image_size=32)
        model = get_model(name, config=cfg)
        x = torch.randn(24, 1, 28, 28); y = torch.randint(0, 10, (24,))
        loader = DataLoader(TensorDataset(x, y), batch_size=8)
        out = tmp_path / "out"
        hist = train_model(model, loader, loader, loader, cfg, torch.device("cpu"),
                           str(out), name)
        weights = out / name / f"{name}_best.pth"
        assert weights.exists()
        spec = json.load(open(spec_path_for(str(weights))))
        assert spec["name"] == name
        assert hist["seed"] == 0 and hist["model_name"] == name
        assert "test_acc" in hist and 0.0 <= hist["test_acc"] <= 1.0
        # Rebuild from checkpoint and confirm it loads
        m2 = load_model_from_checkpoint(str(weights))
        assert m2(X).shape == (2, 10)
