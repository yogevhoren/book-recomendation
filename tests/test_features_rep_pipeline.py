import json
import numpy as np
import pytest

from src.features_representation.pipeline import fit_or_load_all
import src.features_representation.semantic as sem_mod
import src.features_representation.image as img_mod

def _mock_semantic(monkeypatch, dim=8):
    def fake_fit_semantic(df, run_tag, artifacts_root, force_recompute,
                          model_name, batch_size, device, normalize):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(len(df), dim)).astype("float32")
        if normalize:
            n = np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-12)
            Z = (Z / n).astype("float32")
        return Z
    monkeypatch.setattr(sem_mod, "fit_semantic", fake_fit_semantic)

def _mock_image(monkeypatch, dim=4, coverage=1.0):
    def fake_fit_image_embeddings(df, run_tag, artifacts_root, covers_cache_dir,
                                  device, batch_size, force_recompute):
        n = len(df)
        Z = np.ones((n, dim), dtype="float32")
        mask = np.ones(n, dtype=bool)
        if coverage < 1.0:
            k = int(n * coverage)
            mask[:] = False
            mask[:k] = True
        return Z, mask
    monkeypatch.setattr(img_mod, "fit_image_embeddings", fake_fit_image_embeddings)

# def test_fit_all_modalities_no_image(toy_df, artifacts_dir, monkeypatch):
#     _mock_semantic(monkeypatch, dim=12) 
#     reps = fit_or_load_all(
#         toy_df, artifacts_dir,
#         enable_tfidf=True, enable_semantic=True, enable_authors=True, enable_numerics=True, enable_image=False,
#         tfidf_params=dict(blank_non_english=True, title_boost=2, min_df=1, max_df=1.0, force_recompute=True),
#         semantic_params=dict(model_name="BAAI/bge-small-en-v1.5", normalize=True, force_recompute=True),
#     )

#     assert reps.X_tfidf is not None
#     assert reps.X_tfidf.shape[0] == len(toy_df)
#     assert reps.X_tfidf.nnz > 0


#     assert reps.Z_sem is not None and reps.Z_sem.shape == (len(toy_df), 12)
#     norms = np.linalg.norm(reps.Z_sem, axis=1)
#     assert np.allclose(norms[norms > 0], 1.0, atol=1e-3)

#     assert reps.X_auth is not None and reps.X_auth.shape[0] == len(toy_df)
#     rn = np.sqrt((reps.X_auth.multiply(reps.X_auth)).sum(axis=1)).A.ravel()
#     assert np.all(rn[rn > 0] > 0.9)  

#     assert reps.X_num_raw is not None and reps.X_num is not None
#     assert reps.X_num.shape == reps.X_num_raw.shape
#     assert "modern_flag" in reps.num_feature_names
#     assert "pre_1900_flag" in reps.num_feature_names

#     man_path = artifacts_dir / "manifest_all.json"
#     assert man_path.exists()
#     man = json.loads(man_path.read_text(encoding="utf-8"))
#     assert man["enable"]["tfidf"] and man["enable"]["semantic"] and man["enable"]["authors"] and man["enable"]["numerics"]
#     assert man["shapes"]["tfidf"][0] == len(toy_df)
#     assert man["params"]["tfidf"]["run_tag"] == "eda2"
#     assert man["params"]["semantic"]["run_tag"] == "BAAI/bge-small-en-v1.5"

# def test_enable_image_with_mock(toy_df, artifacts_dir, monkeypatch):
#     _mock_semantic(monkeypatch, dim=8)
#     _mock_image(monkeypatch, dim=4, coverage=1.0)

#     reps = fit_or_load_all(
#         toy_df, artifacts_dir,
#         enable_tfidf=False, enable_semantic=True, enable_authors=False, enable_numerics=False, enable_image=True,
#         semantic_params=dict(normalize=True, force_recompute=True),
#         image_params=dict(force_recompute=True),
#     )

#     assert reps.Z_sem is not None and reps.Z_sem.shape == (len(toy_df), 8)
#     assert reps.Z_img is not None and reps.img_mask is not None
#     assert reps.Z_img.shape == (len(toy_df), 4)
#     assert reps.img_mask.mean() == 1.0

# def test_disable_modalities(toy_df, artifacts_dir, monkeypatch):
#     _mock_semantic(monkeypatch, dim=6)
#     reps = fit_or_load_all(
#         toy_df, artifacts_dir,
#         enable_tfidf=False, enable_semantic=False, enable_authors=True, enable_numerics=True, enable_image=False,
#     )

#     assert reps.X_tfidf is None
#     assert reps.Z_sem is None
#     assert reps.X_auth is not None
#     assert reps.X_num is not None


def test_numeric_names_and_shapes(toy_df, artifacts_dir, monkeypatch):
    _mock_semantic(monkeypatch, dim=4)
    reps = fit_or_load_all(
        toy_df, artifacts_dir, 
        enable_tfidf=True, enable_semantic=True, enable_authors=True, enable_numerics=True, enable_image=False, 
        tfidf_params=dict(blank_non_english=True, min_df=1, max_df=1.0)
    )
    assert len(reps.num_feature_names) == reps.X_num.shape[1] == reps.X_num_raw.shape[1]

# def test_reproducible_shapes_same_call(toy_df, artifacts_dir, monkeypatch):
#     _mock_semantic(monkeypatch, dim=7)
#     a = fit_or_load_all(toy_df, artifacts_dir, enable_tfidf=True, enable_semantic=True, enable_authors=True, enable_numerics=True, enable_image=False, 
#                         tfidf_params=dict(blank_non_english=True, title_boost=2, min_df=1, max_df=1.0))
#     b = fit_or_load_all(toy_df, artifacts_dir, enable_tfidf=True, enable_semantic=True, enable_authors=True, enable_numerics=True, enable_image=False, 
#                         tfidf_params=dict(blank_non_english=True, title_boost=2, min_df=1, max_df=1.0))
#     assert a.X_tfidf.shape == b.X_tfidf.shape
#     assert a.Z_sem.shape == b.Z_sem.shape
#     assert a.X_auth.shape == b.X_auth.shape
#     assert a.X_num.shape == b.X_num.shape
