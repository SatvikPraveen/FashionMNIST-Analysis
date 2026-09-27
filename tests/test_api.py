"""
End-to-end REST API test: initialise from a checkpoint + spec written the way
train.py writes them, then classify uploaded images.
Skipped when the serving extras (fastapi, httpx) are not installed.
"""

import io

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from src.models import build_model, ModelSpec  # noqa: E402
from src.models.registry import spec_path_for  # noqa: E402
from src.serving.api import app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    weights = tmp_path / "tinyvgg_best.pth"
    torch.manual_seed(0)
    torch.save(build_model("tinyvgg").state_dict(), weights)
    ModelSpec(name="tinyvgg").save(spec_path_for(str(weights)))
    c = TestClient(app)
    r = c.post("/initialize", params={"model_path": str(weights), "config_path": "config.yaml"})
    assert r.status_code == 200, r.text
    return c, r.json()


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_initialize_rebuilds_architecture_from_spec(client):
    _, body = client
    info = body["model_info"]
    assert info["model_name"] == "tinyvgg"
    assert info["num_parameters"] == 142794


def test_model_info_reports_real_model(client):
    c, _ = client
    info = c.get("/model/info").json()
    assert info["model_name"] == "tinyvgg" and info["input_size"] == 28


@pytest.mark.parametrize("shape", [(28, 28), (60, 40, 3), (100, 100, 4)])
def test_predict_accepts_grayscale_rgb_and_rgba(client, shape):
    c, _ = client
    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    p = c.post("/predict", files={"file": ("x.png", _png(img), "image/png")}).json()
    assert 0 <= p["predicted_class"] < 10
    assert abs(sum(p["all_probabilities"].values()) - 1.0) < 1e-4


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["model_loaded"] is True
