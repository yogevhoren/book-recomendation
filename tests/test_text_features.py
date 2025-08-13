import numpy as np
import pandas as pd
from scipy import sparse
import pytest

from src.features.lexical import build_tfidf_corpus, fit_tfidf, transform_tfidf
from src.features.semantic import build_semantic_corpus, fit_semantic, transform_semantic

def test_lexical_corpus_cleaning_and_blanking():
    df = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["Les Misérables", "A Tale of Two Cities"],
        "description": [
            "Une histoire d’amour à Paris.",               
            "This is an English description about London."
        ],
        "desc_suspected_non_english": [True, False],
    })

    corpus, blank_ratio = build_tfidf_corpus(
        df,
        blank_non_english=True,
        title_boost=2,
    )

    assert isinstance(corpus, list) and len(corpus) == 2

    assert corpus[0].startswith("les miserables les miserables"), "Title should be duplicated for boost"
    assert "paris" not in corpus[0] and "histoire" not in corpus[0]
    assert "é" not in corpus[0]
    text = corpus[1]
    assert "a tale of two cities a tale of two cities" in text, "Boosted title missing"
    for tok in ["english", "description", "london"]:
        assert tok in text
    for sw in [" this ", " is ", " an ", " about ", " the ", " and "]:
        assert sw not in f" {text} "
    assert text == text.lower()
    assert "é" not in text
    assert pytest.approx(blank_ratio, rel=1e-6) == 0.5



def test_lexical_fit_and_transform_roundtrip(tmp_path):
    df = pd.DataFrame({
        "book_id": [11, 12],
        "title": ["The Hobbit", "Dune"],
        "description": ["A hobbit goes on an adventure.", "Desert planet politics and spice."],
        "desc_suspected_non_english": [False, False],
    })
    corpus, _ = build_tfidf_corpus(df, title_boost=2)
    vec, X_fit = fit_tfidf(
        corpus,
        run_tag="unit_bigrams",
        df=df,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=10_000,
        artifacts_root=tmp_path / "tfidf",
    )
    X_tr = transform_tfidf(vec, corpus)
    assert sparse.isspmatrix_csr(X_fit) and sparse.isspmatrix_csr(X_tr)
    assert X_fit.shape == X_tr.shape
    # same vectorizer + same corpus => identical matrix
    assert (X_fit != X_tr).nnz == 0


# --------------------------------
# SEMANTIC (BGE‑M3) — Mandatory
# --------------------------------

def test_semantic_corpus_preserves_punct_and_case():
    df = pd.DataFrame({
        "book_id": [1],
        "title": ["Harry Potter: The Philosopher's Stone"],
        "description": ["Boy wizard meets Hogwarts — and chaos ensues."],
        "desc_suspected_non_english": [False],
    })
    texts = build_semantic_corpus(df)
    assert len(texts) == 1
    assert "Philosopher's Stone" in texts[0]
    assert "—" in texts[0]  # dash preserved
    assert texts[0].splitlines()[1] != ""  # description present


class _MockEncoder:
    def __init__(self):
        self.calls = 0
    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False):
        self.calls += 1
        arr = []
        for t in texts:
            s = sum(ord(c) for c in t)
            arr.append([s, len(t), s % 97, (len(t) % 31)])
        return np.array(arr, dtype=np.float32)


def _mock_factory(model_name, device):
    return _MockEncoder()


def test_semantic_fit_cache_hit_with_mock(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["T1", "T2"],
        "description": ["Desc one.", "Desc TWO!"],
        "desc_suspected_non_english": [False, False],
    })

    # Capture a single shared mock to count calls
    shared = _MockEncoder()
    def factory(_, __):  # ignore args
        return shared

    # 1) fit
    emb1 = fit_semantic(
        df,
        run_tag="bge_m3_unit",
        artifacts_root=tmp_path / "semantic",
        encoder_factory=factory,
        normalize=True,
        batch_size=8,
        model_name="mock",
        device="cpu",
    )
    # 2) fit again (should use cache)
    emb2 = fit_semantic(
        df,
        run_tag="bge_m3_unit",
        artifacts_root=tmp_path / "semantic",
        encoder_factory=factory,
        normalize=True,
        batch_size=8,
        model_name="mock",
        device="cpu",
    )

    assert emb1.shape == emb2.shape == (2, 4)
    # Only one encode() call should have happened across both runs
    assert shared.calls == 1, "Second call should have hit cache and skipped encoding"


def test_semantic_embed_unit_norm_with_mock():
    texts = ["Hello.", "World!!"]
    emb = transform_semantic(
        texts,
        model_name="mock",
        encoder_factory=_mock_factory,
        normalize=True,
        batch_size=4,
        device="cpu",
    )
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_lexical_cache_manifest_respects_config(tmp_path):
    df = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["A A", "B B"],
        "description": ["alpha beta gamma", "delta epsilon zeta"],
        "desc_suspected_non_english": [False, False],
    })
    corpus, _ = build_tfidf_corpus(df)

    # (1,1)
    vec1, X1 = fit_tfidf(
        corpus, run_tag="unit_cfg", df=df, ngram_range=(1, 1), min_df=1, max_df=1.0,
        artifacts_root=tmp_path / "tfidf",
    )
    # (1,2) same run_tag but different config → should refit (manifest mismatch)
    vec2, X2 = fit_tfidf(
        corpus, run_tag="unit_cfg", df=df, ngram_range=(1, 2), min_df=1, max_df=1.0,
        artifacts_root=tmp_path / "tfidf",
    )
    assert X1.shape[1] != X2.shape[1], "Vocab size should change after config change"


def test_semantic_transform_with_mock_is_deterministic():
    texts = ["abc", "def"]
    e1 = transform_semantic(texts, model_name="mock", encoder_factory=_mock_factory, normalize=False)
    e2 = transform_semantic(texts, model_name="mock", encoder_factory=_mock_factory, normalize=False)
    assert np.allclose(e1, e2)

