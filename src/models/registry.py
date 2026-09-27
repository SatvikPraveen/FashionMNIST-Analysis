"""
Single entry point for building any model the project can train.

Two families are supported:

* **Custom CNNs** defined in :mod:`src.models.architectures`
  (``minicnn``, ``tinyvgg``, ``resnet``). These take 1x28x28 input directly.

* **timm backbones** (pretrained or random-init). Any name accepted by
  ``timm.create_model`` works, e.g. ``resnet18``, ``resnet50``,
  ``efficientnet_b0``, ``convnext_tiny``, ``vit_tiny_patch16_224``,
  ``deit_small_patch16_224``. They are wrapped in :class:`TimmClassifier`,
  which upsamples the 28x28 input to ``image_size`` and builds the backbone
  with ``in_chans=1`` so no RGB conversion is needed (timm averages the
  pretrained RGB stem weights when ``in_chans=1``).

Both families expose the same interface: ``model(x)`` where ``x`` is
``(N, 1, 28, 28)`` and the output is ``(N, num_classes)`` logits. This is
what lets the trainer, tuner, evaluator and ensembles treat every model
identically.

Usage:
    from src.models.registry import build_model, model_spec_from_config

    model = build_model("tinyvgg")                       # custom CNN
    model = build_model("resnet18", pretrained=True)     # timm, 224px input
    model = build_model("vit_tiny_patch16_224", image_size=224)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architectures import MiniCNN, TinyVGG, ResNet, BasicBlock

logger = logging.getLogger(__name__)

CUSTOM_MODELS = ("minicnn", "tinyvgg", "resnet")

# Short aliases so config / CLI can say "vit_tiny" instead of the full timm id
TIMM_ALIASES: Dict[str, str] = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "vit": "vit_tiny_patch16_224",
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
}

# Backbones whose patch embedding needs the input size at construction time
_FIXED_SIZE_FAMILIES = ("vit", "deit", "swin", "beit", "eva", "mixer", "pit")


@dataclass
class ModelSpec:
    """Everything needed to rebuild a model for evaluation / serving."""
    name: str
    num_classes: int = 10
    pretrained: bool = False
    image_size: Optional[int] = None      # None -> family default
    in_channels: int = 1
    freeze_backbone: bool = False
    drop_rate: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_timm(self) -> bool:
        return resolve_name(self.name) not in CUSTOM_MODELS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelSpec":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def resolve_name(name: str) -> str:
    """Map CLI/config names to canonical names (custom or timm ids)."""
    n = name.lower().strip()
    if n.startswith("timm:"):
        return n[len("timm:"):]
    if n in CUSTOM_MODELS:
        return n
    return TIMM_ALIASES.get(n, n)


def default_image_size(canonical: str) -> int:
    """Default input resolution per family."""
    if canonical in CUSTOM_MODELS:
        return 28
    if any(fam in canonical for fam in _FIXED_SIZE_FAMILIES):
        return 224
    # CNN backbones are fully convolutional; 64px is a good speed/accuracy
    # trade-off for 28px sources. Override via image_size for 224px transfer.
    return 64


class TimmClassifier(nn.Module):
    """
    Wraps a timm backbone so it consumes 1x28x28 Fashion-MNIST tensors.

    Forward: bilinear-upsample to ``image_size`` -> backbone -> logits.
    """

    def __init__(self, backbone: nn.Module, image_size: int, timm_name: str):
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.timm_name = timm_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return self.backbone(x)

    def get_classifier(self) -> nn.Module:
        return self.backbone.get_classifier()

    def freeze_backbone(self) -> None:
        """Freeze everything except the classification head."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.get_classifier().parameters():
            p.requires_grad = True

    def unfreeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


def _build_timm(canonical: str, spec: ModelSpec, image_size: int) -> nn.Module:
    try:
        import timm
    except ImportError as e:  # pragma: no cover
        raise ImportError("timm is required for pretrained backbones: pip install timm") from e

    kwargs: Dict[str, Any] = dict(
        pretrained=spec.pretrained,
        num_classes=spec.num_classes,
        in_chans=spec.in_channels,
        drop_rate=spec.drop_rate,
    )
    kwargs.update(spec.extra)
    if any(fam in canonical for fam in _FIXED_SIZE_FAMILIES):
        kwargs["img_size"] = image_size

    logger.info(f"Building timm model '{canonical}' "
                f"(pretrained={spec.pretrained}, in_chans={spec.in_channels}, "
                f"image_size={image_size})")
    backbone = timm.create_model(canonical, **kwargs)
    model = TimmClassifier(backbone, image_size=image_size, timm_name=canonical)
    if spec.freeze_backbone:
        model.freeze_backbone()
        logger.info("Backbone frozen; only the classifier head is trainable")
    return model


def build_model(name: str,
                num_classes: int = 10,
                pretrained: bool = False,
                image_size: Optional[int] = None,
                in_channels: int = 1,
                freeze_backbone: bool = False,
                drop_rate: float = 0.0,
                **extra: Any) -> nn.Module:
    """
    Build a model by name.

    Args:
        name: ``minicnn`` / ``tinyvgg`` / ``resnet`` for the custom CNNs, or a
            timm model id / alias (``resnet18``, ``vit_tiny``, ``timm:xyz``).
        num_classes: Output classes.
        pretrained: Load ImageNet weights (timm models only).
        image_size: Input resolution the backbone sees. Custom CNNs are
            always 28. timm CNNs default to 64, ViT-style models to 224.
        in_channels: Input channels (1 for Fashion-MNIST).
        freeze_backbone: Train only the classifier head (timm models only).
        drop_rate: Dropout before the classifier (timm models only).
        **extra: Passed straight to ``timm.create_model``.
    """
    spec = ModelSpec(name=name, num_classes=num_classes, pretrained=pretrained,
                     image_size=image_size, in_channels=in_channels,
                     freeze_backbone=freeze_backbone, drop_rate=drop_rate,
                     extra=dict(extra))
    return build_from_spec(spec)


def build_from_spec(spec: ModelSpec) -> nn.Module:
    canonical = resolve_name(spec.name)

    if canonical == "minicnn":
        return MiniCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    if canonical == "tinyvgg":
        return TinyVGG(in_channels=spec.in_channels, hidden_units=64, num_classes=spec.num_classes)
    if canonical == "resnet":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=spec.num_classes)

    image_size = spec.image_size or default_image_size(canonical)
    return _build_timm(canonical, spec, image_size)


def model_spec_from_config(name: str, config, num_classes: int = 10) -> ModelSpec:
    """
    Build a :class:`ModelSpec` from the project ``Config`` object.

    Reads ``model.pretrained``, ``model.image_size`` (falls back to
    ``data.image_size`` for backwards compatibility),
    ``transfer_learning.freeze_backbone`` and ``model.drop_rate``.
    Custom CNNs ignore the pretrained / size settings.
    """
    canonical = resolve_name(name)
    is_timm = canonical not in CUSTOM_MODELS
    image_size = None
    if is_timm:
        cfg_size = config.get("model.image_size", None)
        image_size = int(cfg_size) if cfg_size else None
    return ModelSpec(
        name=canonical,
        num_classes=num_classes,
        pretrained=bool(config.get("model.pretrained", False)) if is_timm else False,
        image_size=image_size,
        in_channels=1,
        freeze_backbone=bool(config.get("transfer_learning.freeze_backbone", False)) if is_timm else False,
        drop_rate=float(config.get("model.drop_rate", 0.0)) if is_timm else 0.0,
    )


def load_model_from_checkpoint(weights_path: str,
                               spec: Optional[ModelSpec] = None,
                               map_location: str = "cpu") -> nn.Module:
    """
    Rebuild a model from ``<stem>_spec.json`` (or an explicit spec) and load
    its weights. Pretrained download is skipped since weights are restored.
    """
    if spec is None:
        spec_path = spec_path_for(weights_path)
        if not os.path.exists(spec_path):
            raise FileNotFoundError(
                f"No model spec found at {spec_path}; pass spec= explicitly")
        spec = ModelSpec.load(spec_path)
    spec.pretrained = False
    model = build_from_spec(spec)
    state = torch.load(weights_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def spec_path_for(weights_path: str) -> str:
    """``models/x/x_best.pth`` -> ``models/x/x_best_spec.json``."""
    stem, _ = os.path.splitext(weights_path)
    return f"{stem}_spec.json"


def list_models() -> Dict[str, list]:
    """Names the CLI accepts, grouped by family (timm list is the alias set)."""
    return {"custom": list(CUSTOM_MODELS), "timm_aliases": sorted(TIMM_ALIASES)}
