"""
Regression tests for per-sample augmentation and the background fill value.

Before 2026-09-27 the torchvision transforms were called once on the whole
batch tensor, so every image in a batch received the same crop offset, flip
decision and rotation angle, and exposed pixels were filled with 0.0, which
after Fashion-MNIST normalisation is mid-grey instead of black.
"""

import pytest
import torch

from src.data.augmentation import TorchvisionTransforms, AugmentationPipeline, FMNIST_BACKGROUND

BG = FMNIST_BACKGROUND


def _identical_batch(n=64):
    img = torch.full((1, 28, 28), BG)
    img[0, 5:12, 3:9] = 2.0            # asymmetric blob so every transform is visible
    img[0, 18:24, 15:25] = 1.0
    return img.expand(n, 1, 28, 28).clone()


def _n_distinct(x):
    return len({o.numpy().tobytes() for o in x})


@pytest.mark.parametrize("cfg", [
    {"horizontal_flip": True},
    {"random_crop": {"size": 28, "padding": 4}},
    {"rotation": 10},
])
def test_each_image_gets_its_own_draw(cfg):
    torch.manual_seed(0)
    t = TorchvisionTransforms({**cfg, "fill": BG})
    out = t(_identical_batch())
    assert out.shape == (64, 1, 28, 28)
    assert _n_distinct(out) > 1, f"{cfg}: all 64 images received the same transform"


@pytest.mark.parametrize("cfg", [
    {"horizontal_flip": True},
    {"random_crop": {"size": 28, "padding": 4}},
    {"rotation": 10},
])
def test_legacy_mode_reproduces_the_old_bug(cfg):
    torch.manual_seed(0)
    t = TorchvisionTransforms({**cfg, "legacy_batch_mode": True})
    assert _n_distinct(t(_identical_batch())) == 1


def test_flip_rate_is_about_half():
    torch.manual_seed(0)
    x = _identical_batch(2000)
    out = TorchvisionTransforms({"horizontal_flip": True})(x)
    flipped = (out != x).flatten(1).any(1).float().mean().item()
    assert 0.45 < flipped < 0.55


@pytest.mark.parametrize("cfg", [
    {"random_crop": {"size": 28, "padding": 4}},
    {"rotation": 30},
])
def test_exposed_pixels_use_background_not_grey(cfg):
    torch.manual_seed(0)
    x = _identical_batch()
    out = TorchvisionTransforms({**cfg, "fill": BG})(x)
    # Every value is either an original pixel value or the background, never 0.0
    assert not (out == 0.0).any()
    assert torch.isclose(out, torch.tensor(BG)).any()


def test_crop_is_a_pure_translation_of_the_padded_image():
    torch.manual_seed(0)
    x = torch.randn(16, 1, 28, 28)
    t = TorchvisionTransforms({"random_crop": {"size": 28, "padding": 4}, "fill": BG})
    out = t(x)
    padded = torch.nn.functional.pad(x, (4, 4, 4, 4), value=BG)
    for i in range(16):
        windows = padded[i, 0].unfold(0, 28, 1).unfold(1, 28, 1)   # (9, 9, 28, 28)
        match = (windows == out[i, 0]).flatten(2).all(-1)
        assert match.any(), f"image {i} is not a 28x28 window of its padded version"


def test_zero_rotation_and_no_ops_are_identity():
    x = torch.randn(8, 1, 28, 28)
    assert torch.equal(TorchvisionTransforms({})(x), x)
    assert torch.equal(TorchvisionTransforms({"random_crop": {"size": 28, "padding": 0}})(x), x)


def test_single_image_input_still_works():
    x = torch.randn(1, 28, 28)
    out = TorchvisionTransforms({"horizontal_flip": True, "rotation": 5,
                                 "random_crop": {"size": 28, "padding": 2}, "fill": BG})(x)
    assert out.shape == (1, 28, 28)


def test_pipeline_uses_per_sample_transforms():
    torch.manual_seed(0)
    pipe = AugmentationPipeline({"torchvision": {"random_crop": {"size": 28, "padding": 4},
                                                 "horizontal_flip": True, "fill": BG}})
    assert _n_distinct(pipe.apply_sample_augmentations(_identical_batch())) > 10


def test_trainer_passes_background_fill(tmp_path):
    import yaml
    from src.config.settings import Config
    from src.training.trainer import build_augmentation_pipeline
    cfg = {"augmentation": {"enabled": True, "rotation": 10, "horizontal_flip": True,
                            "vertical_flip": False, "random_crop": True, "random_crop_padding": 4,
                            "mixup": False, "cutmix": False}}
    path = tmp_path / "c.yaml"; path.write_text(yaml.safe_dump(cfg))
    tv = build_augmentation_pipeline(Config(str(path))).torchvision_aug
    assert tv.fill == pytest.approx(BG) and not tv.legacy_batch_mode

    cfg["augmentation"].update(fill=0.0, legacy_batch_mode=True)
    path.write_text(yaml.safe_dump(cfg))
    tv = build_augmentation_pipeline(Config(str(path))).torchvision_aug
    assert tv.fill == 0.0 and tv.legacy_batch_mode
