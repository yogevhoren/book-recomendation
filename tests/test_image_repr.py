import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
import requests
import torch


from src.features_representation.image import (
    is_placeholder_url,
    fit_image_embeddings,
    transform_image_embeddings,
    download_covers,
    _letterbox_square,
    _get_dinov2,
)

@pytest.fixture
def tiny_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (20, 30), (120, 10, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class _MockResp:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def mock_requests_ok(monkeypatch, tiny_jpeg_bytes):
    def _get(url, timeout=10):
        return _MockResp(tiny_jpeg_bytes, status_code=200)
    monkeypatch.setattr(requests, "get", _get)
    return _get


@pytest.fixture
def mock_requests_mixed(monkeypatch, tiny_jpeg_bytes):
    def _get(url, timeout=10):
        if "ok" in url:
            return _MockResp(tiny_jpeg_bytes, status_code=200)
        raise RuntimeError("download failed")
    monkeypatch.setattr(requests, "get", _get)
    return _get


class MockEncoder:
    def __init__(self, dim=8):
        self.dim = dim
        self.calls = 0
    def encode(self, batch):
        self.calls += 1
        b = int(batch.shape[0])
        out = np.stack([np.linspace(1, self.dim, self.dim, dtype=np.float32) + i for i in range(b)], axis=0)
        class _T:
            def __init__(self, arr): self._a = arr
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self._a
        return _T(out)
    @property
    def device(self):
        return "cpu"


def _mock_encoder_factory_returning(mock_enc: MockEncoder):
    def factory(model_name: str, device: str | None):
        return mock_enc
    return factory


def test_placeholder_rule_basic():
    assert is_placeholder_url("https://s.gr-assets.com/assets/nophoto/book/111x148-aaa.png")
    assert is_placeholder_url("")  # Empty string instead of None
    assert not is_placeholder_url("http://example.com/cover.jpg")

def test_download_skips_placeholders_and_handles_errors(tmp_path, mock_requests_mixed):
    df = pd.DataFrame({
        "book_id": [1, 2, 3, 4],
        "image_url": [
            "https://s.gr-assets.com/assets/nophoto/book/111x148-aaa.png", 
            "http://example.com/ok1.jpg",  
            "http://example.com/fail.jpg", 
            "http://example.com/ok2.jpg",  
        ],
    })
    out_dir = tmp_path / "covers"
    paths, mask, skipped = download_covers(df, out_dir)
    assert mask.tolist() == [False, True, False, True]
    assert skipped == 1
    assert sum(1 for p in paths if p and Path(p).exists()) == 2


# def test_fit_image_embeddings_row_alignment_and_mask(tmp_path, mock_requests_ok):
#     df = pd.DataFrame({
#         "book_id": [101, 102, 103],
#         "image_url": [
#             "https://s.gr-assets.com/assets/nophoto/book/111x148-aaa.png", 
#             "http://example.com/ok.jpg",   
#             "http://example.com/ok2.jpg",  
#         ],
#     })
#     enc = MockEncoder(dim=16)
#     E, mask = fit_image_embeddings(
#         df,
#         artifacts_root=tmp_path / "image",
#         covers_cache_dir=tmp_path / "image" / "covers",
#         encoder_factory=_mock_encoder_factory_returning(enc),
#         force_recompute=True,
#         batch_size=4,
#     )
#     assert E.shape[0] == len(df) - 1 and E.shape[1] == 16
#     assert mask.tolist() == [False, True, True]
#     norms = np.linalg.norm(E, axis=1)
#     assert norms[1] == pytest.approx(1.0, rel=1e-5)
#     covers_dir = tmp_path / "image" / "covers"
#     assert any(covers_dir.iterdir()), "Downloaded covers not found on disk"


def test_fit_image_embeddings_cache_hit(tmp_path, mock_requests_ok):
    df = pd.DataFrame({
        "book_id": [1, 2],
        "image_url": ["http://example.com/ok.jpg", "http://example.com/ok2.jpg"],
    })
    enc = MockEncoder(dim=8)
    E1, mask1 = fit_image_embeddings(
        df,
        artifacts_root=tmp_path / "image",
        covers_cache_dir=tmp_path / "image" / "covers",
        encoder_factory=_mock_encoder_factory_returning(enc),
        force_recompute=True,
        batch_size=4,
    )
    calls_after_first = enc.calls

    E2, mask2 = fit_image_embeddings(
        df,
        artifacts_root=tmp_path / "image",
        covers_cache_dir=tmp_path / "image" / "covers",
        encoder_factory=_mock_encoder_factory_returning(enc),
        force_recompute=False,
        batch_size=4,
    )
    assert enc.calls == calls_after_first, "Cache hit should avoid re-encoding"
    assert np.allclose(E1, E2)
    assert np.array_equal(mask1, mask2)

    man_path = tmp_path / "image" / "dinov2_vits14" / "manifest.json"
    assert man_path.exists()
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert man["shape"] == [2, 8]
    assert man["pct_missing"] <= 100.0
    assert man["book_ids"] == [1, 2]


def test_transform_image_embeddings_with_paths(tmp_path):
    def _save(img_size, name):
        img = Image.new("RGB", img_size, (50, 200, 30))
        fp = tmp_path / f"{name}.jpg"
        img.save(fp, format="JPEG")
        return str(fp)

    p1 = _save((30, 30), "a")
    p2 = _save((20, 40), "b")
    missing = str(tmp_path / "nope.jpg")

    enc = MockEncoder(dim=12)
    E = transform_image_embeddings(
        [p1, missing, p2],
        encoder_factory=_mock_encoder_factory_returning(enc),
        batch_size=3,
    )
    assert E.shape == (3, 12)
    assert np.allclose(E[1], 0.0)
    assert np.isclose(np.linalg.norm(E[0]), 1.0, rtol=1e-5)
    assert np.isclose(np.linalg.norm(E[2]), 1.0, rtol=1e-5)


def test_image_resizing():
    img = Image.new("RGB", (100, 200), (255, 0, 0))
    resized_tensor = _letterbox_square(img, target_size=(518, 518))  # Corrected argument
    assert resized_tensor.shape == (3, 518, 518), "Tensor does not have the expected shape"

def test_get_dinov2_input_validation():
    with pytest.raises(ValueError, match="Input dimensions are incorrect"):
        _get_dinov2("vit_small_patch14_dinov2", device="cpu").encode(torch.empty((1, 3, 500, 500)))  # Invalid shape


