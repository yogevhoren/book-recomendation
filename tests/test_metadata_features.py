import numpy as np
import pandas as pd
from scipy import sparse

from src.features.metadata import (
    fit_authors,
    transform_authors,
    fit_numeric_scalers,
    transform_numerics,
)

def test_b1_canonicalization_merges_flipped_names():
    df = pd.DataFrame({
        "book_id": [1, 2],
        "author_list": [["Rowling, J.K."], ["J.K. Rowling"]],
    })
    mlb, iaf = fit_authors(df, use_canonical=True, artifacts_dir="artifacts/metadata")
    X = transform_authors(df, mlb, iaf, use_canonical=True).toarray()
    assert X.shape == (2, X.shape[1])
    assert X[0].sum() > 0 and X[1].sum() > 0
    assert np.array_equal((X[0] > 0).astype(int), (X[1] > 0).astype(int)), \
        "Flipped author variants should activate the exact same column"


def test_b1_iaf_popular_author_has_lower_weight_than_rare():
    df = pd.DataFrame({
        "book_id": [10, 11, 12, 13, 14, 15, 16],
        "author_list": [
            ["Popular, Paul"], ["Popular, Paul"], ["Popular, Paul"],
            ["Popular, Paul"], ["Popular, Paul"],
            ["Rare, Ria"], ["Rare, Ria"],
        ],
    })
    mlb, iaf = fit_authors(df, use_canonical=True, artifacts_dir="artifacts/metadata")
    classes = list(mlb.classes_)
    idx_pop = classes.index("Paul Popular") 
    idx_rare = classes.index("Ria Rare")
    assert iaf[idx_pop] < iaf[idx_rare], "IAF should down-weight popular author vs rare author"

    Xw = transform_authors(df, mlb, iaf, use_canonical=True)
    assert sparse.isspmatrix_csr(Xw)
    assert Xw.shape[0] == len(df)


def test_b1_transform_row_alignment():
    df = pd.DataFrame({
        "book_id": [1, 2, 3],
        "author_list": [["A, A"], ["B, B"], ["A, A"]],
    })
    mlb, iaf = fit_authors(df, use_canonical=True, artifacts_dir="artifacts/metadata")
    X = transform_authors(df, mlb, iaf, use_canonical=True).toarray()
    assert np.array_equal(X[0], X[2])
    assert not np.array_equal(X[0], X[1])

def test_b2_numeric_scaling_and_flags_core():
    df = pd.DataFrame({
        "book_id": [1, 2, 3, 4],
        "description": [
            "short text",
            "this is a somewhat longer description",
            "x " * 50,
            "only one",
        ],
        "author_list": [
            ["Last, First"],
            ["One, Author", "Two, Author"],
            [],
            ["Solo, Person"],
        ],
        "average_rating": [3.0, 4.0, 4.77, 2.47],
        "original_publication_year": [-450, 1995, 2030, 2000], 
    })

    cfg = fit_numeric_scalers(df, artifacts_dir="artifacts/metadata")
    X_num = transform_numerics(df, cfg)

    assert X_num.ndim == 2 and X_num.shape[0] == 4
    ncols = X_num.shape[1]
    assert ncols == 8, f"Unexpected numeric width: {ncols}"

    rating_minmax = X_num[:, 4]
    assert np.all(rating_minmax >= 0.0) and np.all(rating_minmax <= 1.0)
    year_norm = X_num[:, 5]
    assert np.all(year_norm >= 0.0) and np.all(year_norm <= 1.0), "year_norm must be in [0,1]"

    modern_flag = X_num[:, 6].astype(int).tolist()
    assert modern_flag == [0, 0, 1, 1]

    pre_1900_flag = X_num[:, 7].astype(int).tolist()
    assert pre_1900_flag == [1, 0, 0, 0]

    desc_len_log1p = X_num[:, 0]
    assert desc_len_log1p[2] > desc_len_log1p[1] > desc_len_log1p[0]

    author_count_int = X_num[:, 2].tolist()
    assert author_count_int == [1.0, 2.0, 0.0, 1.0]


def test_b2_numeric_cfg_persistence_manifest_keys():
    df = pd.DataFrame({
        "book_id": [1],
        "description": ["a b c"],
        "author_list": [["Last, First"]],
        "average_rating": [4.0],
        "original_publication_year": [2001],
    })
    cfg = fit_numeric_scalers(df, artifacts_dir="artifacts/metadata")
    assert "numeric_feature_names" in cfg
    assert "rating_minmax" in cfg and {"min", "max"}.issubset(cfg["rating_minmax"].keys())
    assert "year_bounds" in cfg and "clip_max" in cfg["year_bounds"]
    X_num = transform_numerics(df, cfg)
    assert X_num.shape[0] == 1