import numpy as np
import pandas as pd
from src.recsys.metrics import uniqueness_from_agreement, weight_scorecard, deltas_over_threshold, diversity_check

def test_diversity_check_known_values():
    df = pd.DataFrame({
        "authors": ["A", "A", "B", "B"],
        "original_publication_year": [2000, 2005, 2010, 2015]
    })
    mat = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0]
    ])
    hhi_scores, year_spreads = diversity_check(mat, df, k=1)
    assert np.allclose(hhi_scores, np.ones(len(df)))
    assert np.allclose(year_spreads, np.zeros(len(df)))

def test_diversity_check_diverse_neighbors():
    df = pd.DataFrame({
        "authors": ["A", "B", "C"],
        "original_publication_year": [2000, 2001, 2002]
    })
    mat = np.ones((3, 3)) - np.eye(3)
    hhi_scores, year_spreads = diversity_check(mat, df, k=2)
    assert np.allclose(hhi_scores, np.full(len(df), 0.5))
    assert np.allclose(year_spreads, np.array([1, 2, 1]))

def test_uniqueness_from_agreement():
    df = pd.DataFrame({
        "pair":["semantic_tfidf","semantic_authors","tfidf_authors"],
        "jaccard@10":[0.6,0.4,0.3],
        "kendall_tau@10":[0.5,0.2,0.1]
    })
    u = uniqueness_from_agreement(df, anchor="semantic")
    assert "tfidf" in u and "authors" in u
    assert 0.0 <= u["tfidf"] <= 1.0

def test_weight_scorecard_and_deltas():
    baseline = {"semantic":0.45,"tfidf":0.25,"authors":0.10,"numeric":0.10,"image":0.10}
    stability = {"semantic":0.80,"tfidf":0.70,"authors":0.50,"numeric":0.60,"image":0.40}
    separation = {"semantic":0.85,"tfidf":0.65,"authors":0.55,"numeric":0.50,"image":0.45}
    uniqueness = {"tfidf":0.30,"authors":0.45,"numeric":0.20,"image":0.25}
    rec, df = weight_scorecard(baseline, stability, separation, uniqueness, include_image=True, image_cap=0.05, other_cap=0.5)
    assert abs(sum(rec.values())-1.0) < 1e-6
    assert set(df.columns)=={"track","baseline","recommended","stability","separation","uniqueness","cap"}
    deltas = deltas_over_threshold(df, thr=0.05)
    assert isinstance(deltas, pd.DataFrame)
    assert set(deltas.columns).issuperset({"track","delta"})
