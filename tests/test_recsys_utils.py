import numpy as np
import pandas as pd
from scipy import sparse
from src.recsys.utils import (
    cosine_sim_dense, cosine_sim_csr, topk_sim,
    topk_indices_from_track, jaccard_at_k, kendall_tau_approx, hhi,
    sample_negatives, pos_neg_index_sets, compute_norm_outlier_share,
    umap_embed, resample_stability, tfidf_top_terms, nearest_table, enforce_caps
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def test_cosine_sim_dense_and_csr_agree():
    X = np.array([[1.0, 0.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    X_norm = normalize(X, norm="l2")
    S_dense = cosine_sim_dense(X_norm, X_norm)
    S_csr = cosine_sim_csr(sparse.csr_matrix(X_norm)).toarray()
    assert np.allclose(S_dense, S_csr, atol=1e-6)


def test_topk_indices_from_track():
    X = np.array([[1,0],[1,0.1],[0,1]],dtype=float)
    idx = topk_indices_from_track(X, k=2)
    assert idx.shape == (3,2)
    assert all(i not in row for i,row in enumerate(idx))

def test_jaccard_and_kendall():
    a = np.array([[1,2,3,4],[4,3,2,1]])
    b = np.array([[1,2,5,6],[1,2,3,4]])
    j = jaccard_at_k(a[:,:3],b[:,:3])
    t = kendall_tau_approx(a[:,:3],b[:,:3])
    assert np.isclose(j[0],2/3)
    assert t.shape==(2,)

def test_hhi():
    x = [1,1,2,2,2,3]
    v = hhi(x)
    assert 0 < v < 1

def test_sample_negatives():
    n,k=10,3
    excl = {0:[1,2], 1:[0]}
    res = sample_negatives(n,k,excl)
    assert len(res)==n
    assert len(res[0])<=k
    assert all(x not in excl.get(0,[]) and x!=0 for x in res[0])

def test_pos_neg_index_sets():
    df = pd.DataFrame({
        "book_id":[1,2,3,4],
        "author_list":[["A"],["A"],["B"],["C"]],
        "desc_hash":["h1","h1","h2","h3"]
    })
    pos,neg = pos_neg_index_sets(df,k=2)
    assert any(len(p)>0 for p in pos)
    assert len(neg)==4

def test_compute_norm_outlier_share():
    X = np.vstack([np.random.normal(0,1,(98,8)), np.ones((2,8))*100])
    s = compute_norm_outlier_share(X, mad_thresh=5)
    assert s > 0

def test_umap_embed_shape():
    X = np.random.RandomState(0).randn(50,8)
    E = umap_embed(X, n_neighbors=10, random_state=42)
    assert E.shape==(50,2)

def test_resample_stability_small():
    rng = np.random.RandomState(0)
    X = rng.randn(60,16)
    v = resample_stability(X, k=10, trials=2, frac=0.6, seed=123)
    assert 0.0 <= v <= 1.0

def test_tfidf_top_terms_and_nearest_table():
    docs = ["red apple fruit","green apple tree","blue ocean water","mountain hiking trail"]
    vec = TfidfVectorizer().fit(docs)
    terms = tfidf_top_terms(vec, "red apple", top=3)
    assert len(terms)>0
    track = vec.transform(docs).toarray()
    df = pd.DataFrame({
        "book_id":[10,11,12,13],
        "title":["a","b","c","d"],
        "author_list":[["A"],["A"],["B"],["C"]],
        "original_publication_year":[2000,2001,1990,1980],
        "average_rating":[4.0,3.5,4.2,3.8]
    })
    tab = nearest_table(track, df, seed_idx=0, top=3)
    assert len(tab)==3
    assert "cosine" in tab.columns

def test_enforce_caps():
    df = pd.DataFrame({
        "book_id":[0,1,2,3,4,5],
        "author_list":[["A"],["A"],["A"],["B"],["B"],["C"]],
        "desc_hash":["h1","h1","h2","h3","h3","h4"]
    }).set_index("book_id")
    ranked = [0,1,2,3,4,5]
    kept = enforce_caps(ranked, df, max_per_author=2, max_per_desc=1)
    assert len(kept) <= len(ranked)
    assert sum(1 for i in kept if df.loc[i,"author_list"][0]=="A") <= 2
    assert sum(1 for i in kept if df.loc[i,"desc_hash"]=="h1") <= 1
