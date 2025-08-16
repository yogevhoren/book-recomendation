from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import pickle 


log = logging.getLogger(__name__)

_punct_num_re = re.compile(r"[^a-z\s]+") 
_ws_re = re.compile(r"\s+")

def _ascii_fold(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _clean_title_for_tfidf(title: str) -> str:
    s = _ascii_fold(title).lower()
    s = _punct_num_re.sub(" ", s)
    return _ws_re.sub(" ", s).strip()

def _clean_desc_for_tfidf(desc: str, *, stopwords: set[str]) -> str:
    s = _ascii_fold(desc).lower()
    s = _punct_num_re.sub(" ", s)
    toks = [t for t in s.split() if t and t not in stopwords]
    return " ".join(toks)

def _df_hash_for_tfidf(df: pd.DataFrame) -> str:
    need = ["book_id", "title", "description", "desc_suspected_non_english"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"_df_hash_for_tfidf missing column: {c}")
    joined = (
        df["book_id"].astype(str).fillna("")
        + "\u241F"
        + df["title"].astype(str).fillna("")
        + "\u241F"
        + df["description"].astype(str).fillna("")
        + "\u241F"
        + df["desc_suspected_non_english"].astype(str)
    ).tolist()
    h = sha256()
    for row in joined:
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _corpus_hash(texts: Sequence[str]) -> str:
    h = sha256()
    for t in texts:
        h.update((t or "").encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

@dataclass
class TfidfArtifacts:
    out_dir: Path
    vec_path: Path
    X_path: Path
    manifest_path: Path

    @staticmethod
    def in_dir(base: Path) -> "TfidfArtifacts":
        base.mkdir(parents=True, exist_ok=True)
        return TfidfArtifacts(
            out_dir=base,
            vec_path=base / "vectorizer.pkl",
            X_path=base / "X.npz",
            manifest_path=base / "manifest.json",
        )

    def save_manifest(
        self,
        *,
        book_ids: Sequence[int],
        df_hash: str,
        corpus_hash: str,
        cfg: dict,
        vectorizer_vocab_size: int,
        nnz: int,
        shape: Tuple[int, int],
        blanked_desc_ratio: float,
    ) -> None:
        man = {
            "book_ids": list(map(int, book_ids)),
            "df_hash": df_hash,
            "corpus_hash": corpus_hash,
            "config": cfg,
            "vocab_size": int(vectorizer_vocab_size),
            "nnz": int(nnz),
            "shape": list(map(int, shape)),
            "blanked_desc_ratio": float(blanked_desc_ratio),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)

def build_tfidf_corpus(
    df: pd.DataFrame,
    *,
    blank_non_english: bool = True,
    title_boost: int = 2,
) -> Tuple[List[str], float]:
    need = ["book_id", "title", "description", "desc_suspected_non_english"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"build_tfidf_corpus missing column: {c}")

    sw = set(ENGLISH_STOP_WORDS)

    desc_flag = df["desc_suspected_non_english"].fillna(False).astype(bool).to_numpy()
    blanked = 0
    texts: List[str] = []
    for title, desc, is_non_en in zip(df["title"].astype(str), df["description"].astype(str), desc_flag):
        t = _clean_title_for_tfidf(title)
        if blank_non_english and is_non_en:
            d = ""  
            blanked += 1
        else:
            d = _clean_desc_for_tfidf(desc, stopwords=sw)
        boosted_title = (t + " ") * max(1, title_boost)
        texts.append((boosted_title + d).strip())

    ratio = blanked / max(1, len(df))
    log.info(
        "TF‑IDF corpus built: rows=%d | title_boost=%d | blanked_non_en=%.1f%%",
        len(texts), title_boost, 100 * ratio
    )
    return texts, ratio


def fit_tfidf(
    corpus: Sequence[str],
    *,
    run_tag: str,
    df: pd.DataFrame,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int | float = 5,
    max_df: float = 0.90,
    max_features: int = 120_000,
    sublinear_tf: bool = True,
    norm: str = "l2",
    artifacts_root: Path | str = "artifacts/tfidf",
    force_recompute: bool = False,
) -> Tuple[TfidfVectorizer, sparse.csr_matrix]:
    outdir = Path(artifacts_root) / run_tag
    arts = TfidfArtifacts.in_dir(outdir)

    cfg = dict(
        run_tag=run_tag,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm=norm,
    )

    df_hash = _df_hash_for_tfidf(df)
    c_hash = _corpus_hash(corpus)

    if not force_recompute and arts.manifest_path.exists() and arts.vec_path.exists() and arts.X_path.exists():
        try:
            with open(arts.manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if man.get("df_hash") == df_hash and man.get("corpus_hash") == c_hash and man.get("config") == cfg:
                log.info("TF‑IDF [%s] cache hit → loading from %s", run_tag, outdir)
                with open(arts.vec_path, "rb") as f:
                    vec: TfidfVectorizer = pickle.load(f) 
                X = load_npz(arts.X_path).tocsr()
                return vec, X
        except Exception as e:
            log.warning("TF‑IDF [%s] manifest present but load failed (%s). Recomputing.", run_tag, e)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        norm=norm,
        dtype=np.float32,
        token_pattern=r"(?u)\b\w+\b", 
    )
    X = vectorizer.fit_transform(corpus).tocsr()

    with open(arts.vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    save_npz(arts.X_path, X)

    arts.save_manifest(
        book_ids=df["book_id"].tolist(),
        df_hash=df_hash,
        corpus_hash=c_hash,
        cfg=cfg,
        vectorizer_vocab_size=len(vectorizer.vocabulary_),
        nnz=int(X.nnz),
        shape=X.shape,
        blanked_desc_ratio=float(np.nan), 
    )
    log.info(
        "TF‑IDF [%s] fit complete: shape=%s | vocab=%d | nnz=%d",
        run_tag, X.shape, len(vectorizer.vocabulary_), int(X.nnz)
    )
    return vectorizer, X


def transform_tfidf(vectorizer: TfidfVectorizer, corpus: Sequence[str]) -> sparse.csr_matrix:
    X = vectorizer.transform(corpus).tocsr()
    return X
