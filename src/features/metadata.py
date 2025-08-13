from __future__ import annotations

import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer

log = logging.getLogger(__name__)

_METADATA_ARTS_DIR = Path("artifacts/metadata")

_ws_re = re.compile(r"\s+")

def _collapse_ws(s: str) -> str:
    return _ws_re.sub(" ", s).strip()

def _is_single_person_last_comma_first(token: str) -> bool:
    if not isinstance(token, str):
        return False
    t = token.strip()
    if t.count(",") != 1:
        return False
    low = t.lower()
    if " & " in low or " and " in low:
        return False
    return True

def _canon_person(token: str) -> str:
    if not _is_single_person_last_comma_first(token):
        return _collapse_ws(token if isinstance(token, str) else "")
    last, first = [part.strip() for part in token.split(",", 1)]
    if not last or not first:
        return _collapse_ws(token)
    return _collapse_ws(f"{first} {last}")


def canonicalize_authors(author_list: Sequence[str]) -> List[str]:
    if not isinstance(author_list, (list, tuple)):
        return []
    out = []
    for tok in author_list:
        out.append(_canon_person(tok))
    return out

def _df_hash_for_authors(df: pd.DataFrame) -> str:
    if "book_id" not in df.columns:
        raise ValueError("Required column missing for authors hash: book_id")

    if "authors" in df.columns:
        left = df["authors"].astype(str).fillna("")
    elif "author_list" in df.columns:
        left = df["author_list"].apply(
            lambda xs: "|".join([s.strip() for s in xs]) if isinstance(xs, (list, tuple)) else ""
        )
    else:
        raise ValueError("Required column missing for authors hash: 'authors' or 'author_list'")

    joined = (df["book_id"].astype(str).fillna("") + "\u241F" + left).tolist()
    h = sha256()
    for row in joined:
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class AuthorsArtifacts:
    out_dir: Path
    mlb_path: Path
    iaf_path: Path
    manifest_path: Path

    @staticmethod
    def in_dir(base: Path) -> "AuthorsArtifacts":
        base.mkdir(parents=True, exist_ok=True)
        return AuthorsArtifacts(
            out_dir=base,
            mlb_path=base / "authors_mlb.pkl",
            iaf_path=base / "authors_iaf.npy",
            manifest_path=base / "authors_manifest.json",
        )

    def save(self, *, mlb: MultiLabelBinarizer, iaf: np.ndarray,
             book_ids: Sequence[int], vocab: Sequence[str],
             df_hash: str, pct_canon_changed: float) -> None:
        with open(self.mlb_path, "wb") as f:
            pickle.dump(mlb, f)
        np.save(self.iaf_path, iaf)
        manifest = {
            "book_ids": list(map(int, book_ids)),
            "vocab": list(vocab),
            "df_hash": df_hash,
            "percent_canonicalized_changed": float(pct_canon_changed),
            "n_books": int(len(book_ids)),
            "n_authors": int(len(vocab)),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

def _percent_changed(before: List[List[str]], after: List[List[str]]) -> float:
    changed = 0
    total = 0
    for b, a in zip(before, after):
        if b != a:
            changed += 1
        total += 1
    return (changed / total) if total else 0.0

def _collect_top_examples(before: List[List[str]], after: List[List[str]], k: int = 10) -> List[Tuple[List[str], List[str]]]:
    ex = []
    for b, a in zip(before, after):
        if b != a:
            ex.append((b, a))
        if len(ex) >= k:
            break
    return ex


def fit_authors(
    df: pd.DataFrame,
    *,
    use_canonical: bool = True,
    artifacts_dir: Path | str = _METADATA_ARTS_DIR,
) -> Tuple[MultiLabelBinarizer, np.ndarray]:
    if "book_id" not in df.columns or "author_list" not in df.columns:
        raise ValueError("fit_authors expects columns: 'book_id', 'author_list' (from cleaning).")

    df_hash = _df_hash_for_authors(df)

    raw_lists: List[List[str]] = [
        [s.strip() for s in (lst if isinstance(lst, (list, tuple)) else []) if isinstance(s, str)]
        for lst in df["author_list"].tolist()
    ]
    if use_canonical:
        can_lists = [canonicalize_authors(lst) for lst in raw_lists]
    else:
        can_lists = raw_lists

    pct_changed = _percent_changed(raw_lists, can_lists)
    if use_canonical:
        examples = _collect_top_examples(raw_lists, can_lists, k=10)
        if examples:
            log.info("Author canonicalization touched %.1f%% of rows. Examples (before -> after):", pct_changed * 100)
            for b, a in examples:
                log.info("  %s  ->  %s", b, a)
        else:
            log.info("Author canonicalization touched %.1f%% of rows. No examples to show.", pct_changed * 100)

    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform(can_lists) 
    n_docs, n_authors = X.shape
    if n_authors == 0:
        iaf = np.zeros((0,), dtype=np.float32)
        log.warning("No authors found while fitting ML-Binarizer.")
    else:
        df_per_author = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.int64) 
        iaf = (np.log((n_docs + 1) / (df_per_author + 1)) + 1.0).astype(np.float32) 

    arts = AuthorsArtifacts.in_dir(Path(artifacts_dir))
    arts.save(
        mlb=mlb,
        iaf=iaf,
        book_ids=df["book_id"].tolist(),
        vocab=mlb.classes_.tolist(),
        df_hash=df_hash,
        pct_canon_changed=pct_changed,
    )
    log.info("Saved authors artifacts to %s (n_docs=%d, n_authors=%d).", arts.out_dir, n_docs, n_authors)
    return mlb, iaf


def transform_authors(
    df: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    iaf: np.ndarray,
    *,
    use_canonical: bool = True,
) -> sparse.csr_matrix:
    if "author_list" not in df.columns:
        raise ValueError("transform_authors expects column: 'author_list' (from cleaning).")

    raw_lists: List[List[str]] = [
        [s.strip() for s in (lst if isinstance(lst, (list, tuple)) else []) if isinstance(s, str)]
        for lst in df["author_list"].tolist()
    ]
    lists = [canonicalize_authors(lst) for lst in raw_lists] if use_canonical else raw_lists

    X = mlb.transform(lists).tocsr() 
    if iaf is not None and iaf.size:
        X = X.multiply(iaf)
    return X.tocsr()

def _df_hash_for_numeric(df: pd.DataFrame) -> str:
    required = ["book_id", "original_publication_year", "average_rating", "description", "author_list"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing for numeric hash: {c}")

    series = []
    series.append(df["book_id"].astype(str).fillna(""))
    year = pd.to_numeric(df["original_publication_year"], errors="coerce").fillna(-1).astype(int).astype(str)
    rating = pd.to_numeric(df["average_rating"], errors="coerce").fillna(-1.0).round(6).astype(str)
    desc_len = df["description"].astype(str).map(lambda s: len(s.split()))
    alen = df["author_list"].map(lambda xs: len(xs) if isinstance(xs, (list, tuple)) else 0).astype(int).astype(str)

    series.extend([year, rating, desc_len.astype(str), alen])

    joined = (series[0] + "\u241F" + series[1] + "\u241F" + series[2] + "\u241F" + series[3] + "\u241F" + series[4]).tolist()
    h = sha256()
    for row in joined:
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class NumericArtifacts:
    out_dir: Path
    scalers_path: Path
    manifest_path: Path

    @staticmethod
    def in_dir(base: Path) -> "NumericArtifacts":
        base.mkdir(parents=True, exist_ok=True)
        return NumericArtifacts(
            out_dir=base,
            scalers_path=base / "numeric_scalers.pkl",
            manifest_path=base / "numeric_manifest.json",
        )

    def save(self, *, cfg: dict, book_ids: Sequence[int], df_hash: str) -> None:
        with open(self.scalers_path, "wb") as f:
            pickle.dump(cfg, f)
        manifest = {
            "book_ids": list(map(int, book_ids)),
            "df_hash": df_hash,
            "n_books": int(len(book_ids)),
            "numeric_feature_names": cfg.get("numeric_feature_names", []),
            "medians": cfg.get("medians", {}),
            "rating_minmax": cfg.get("rating_minmax", {}),
            "year_bounds": cfg.get("year_bounds", {}),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def _safe_len_words(s: str) -> int:
    if not isinstance(s, str):
        return 0
    return len(s.split())


def _safe_author_count(xs) -> int:
    if isinstance(xs, (list, tuple)):
        return sum(1 for v in xs if isinstance(v, str) and v.strip() != "")
    return 0


def fit_numeric_scalers(df: pd.DataFrame, *, artifacts_dir: Path | str = _METADATA_ARTS_DIR) -> dict:
    needed = ["book_id", "description", "author_list", "average_rating", "original_publication_year"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"fit_numeric_scalers expects column: {c}")

    df_hash = _df_hash_for_numeric(df)

    desc_len_words = df["description"].astype(str).map(_safe_len_words)
    author_count = df["author_list"].map(_safe_author_count)

    desc_len_median = float(desc_len_words.median())
    author_count_median = float(author_count.median())

    rating = pd.to_numeric(df["average_rating"], errors="coerce")
    r_min = float(rating.min(skipna=True)) if rating.notna().any() else 0.0
    r_max = float(rating.max(skipna=True)) if rating.notna().any() else 1.0
    if r_max <= r_min:
        r_min, r_max = 0.0, 1.0

    year_raw = pd.to_numeric(df["original_publication_year"], errors="coerce")
    year_cap = year_raw.clip(upper=2026)
    y_min = float(year_cap.min(skipna=True)) if year_cap.notna().any() else -float("inf")
    y_max = float(year_cap.max(skipna=True)) if year_cap.notna().any() else 2026.0
    if y_max <= y_min:
        y_min, y_max = y_min, y_min + 1.0  


    cfg = {
        "medians": {
            "desc_len_words": desc_len_median,
            "author_count": author_count_median,
        },
        "rating_minmax": {"min": r_min, "max": r_max},
        "year_bounds": {"min": y_min, "max": y_max, "clip_max": 2026},
        "numeric_feature_names": [
            "desc_len_log1p",
            "desc_len_ge_median",
            "author_count_int",
            "author_count_ge_median",
            "average_rating_minmax",
            "year_norm",
            "modern_flag",
            "pre_1900_flag",
        ],
        "df_hash": df_hash,
    }

    arts = NumericArtifacts.in_dir(Path(artifacts_dir))
    arts.save(cfg=cfg, book_ids=df["book_id"].tolist(), df_hash=df_hash)
    log.info(
        "Saved numeric scalers to %s | medians(desc_len=%.1f, author_count=%.1f) | "
        "rating[min=%.2f, max=%.2f] | year[min=%.0f, max=%.0f]",
        arts.out_dir, desc_len_median, author_count_median, r_min, r_max, y_min, y_max)
    return cfg


def transform_numerics(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    needed = ["description", "author_list", "average_rating", "original_publication_year"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"transform_numerics expects column: {c}")

    med = cfg["medians"]
    rmin, rmax = cfg["rating_minmax"]["min"], cfg["rating_minmax"]["max"]
    ymin, ymax = cfg["year_bounds"]["min"], cfg["year_bounds"]["max"]
    desc_len_words = df["description"].astype(str).map(_safe_len_words).astype(float)
    author_count = df["author_list"].map(_safe_author_count).astype(float)

    desc_len_log1p = np.log1p(desc_len_words.values)
    desc_len_ge_med = (desc_len_words.values >= med["desc_len_words"]).astype(np.float32)

    author_count_int = author_count.values
    author_count_ge_med = (author_count.values >= med["author_count"]).astype(np.float32)

    rating = pd.to_numeric(df["average_rating"], errors="coerce").astype(float)
    if rmax > rmin:
        rating_minmax = ((rating - rmin) / (rmax - rmin)).clip(lower=0.0, upper=1.0).fillna(0.0).values
    else:
        rating_minmax = np.zeros(len(df), dtype=np.float32)

    year_raw = pd.to_numeric(df["original_publication_year"], errors="coerce").astype(float)
    year_cap = year_raw.clip(upper=2026) 
    if ymax > ymin:
        year_norm = ((year_cap - ymin) / (ymax - ymin)).clip(lower=0.0, upper=1.0).fillna(0.0).values
    else:
        year_norm = np.zeros(len(df), dtype=np.float32)


    modern_flag = (year_raw >= 2000).fillna(False).astype(np.float32).values
    pre_1900_flag = (year_raw < 1900).fillna(False).astype(np.float32).values 

    X_num = np.column_stack([
        desc_len_log1p.astype(np.float32),
        desc_len_ge_med,
        author_count_int.astype(np.float32),
        author_count_ge_med,
        rating_minmax.astype(np.float32),
        year_norm.astype(np.float32),
        modern_flag.astype(np.float32),
        pre_1900_flag.astype(np.float32),
    ])
    log.info("Numeric transform: X_num shape=%s ", X_num.shape)
    return X_num
