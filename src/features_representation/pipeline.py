from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize
from src.features_representation import lexical, metadata, semantic, image

log = logging.getLogger(__name__)

# --- add at top (after imports) ---
from typing import Tuple
try:
    from src.clean.pipeline import clean_books_dataset
except Exception as _e:
    clean_books_dataset = None  # we'll fail clearly if we can't clean


def ensure_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    need_any = [
        "author_list",
        "desc_suspected_non_english",
        "desc_hash",
        "desc_group_size",
        "desc_is_shared",
    ]
    if all(c in df.columns for c in need_any):
        return df

    if clean_books_dataset is None:
        missing = [c for c in need_any if c not in df.columns]
        raise RuntimeError(
            f"Input df missing required cleaned columns {missing} and "
            f"src.clean.pipeline.clean_books_dataset is unavailable to call."
        )

    df_clean = clean_books_dataset(df, drop_language_col=True)
    missing = [c for c in need_any if c not in df_clean.columns]
    if missing:
        raise RuntimeError(
            f"Cleaner did not produce required columns: {missing}"
        )
    return df_clean


@dataclass
class Reps:
    X_tfidf: Optional[object]
    tfidf_vec: Optional[object]
    Z_sem: Optional[np.ndarray]
    X_auth: Optional[object]
    authors_mlb: Optional[object]
    authors_iaf: Optional[np.ndarray]
    X_num_raw: Optional[np.ndarray]
    X_num: Optional[np.ndarray]
    num_feature_names: list[str]
    Z_img: Optional[np.ndarray]
    img_mask: Optional[np.ndarray]
    manifest: dict

def _safe(d: dict, k: str, default):
    return d[k] if k in d and d[k] is not None else default

def fit_or_load_all(
    df,
    artifacts_root: Path,
    *,
    run_tag_tfidf: str = "eda2",
    run_tag_sem: str = "bge-small-en-v1.5",
    run_tag_img: str = "dinov2_vits14",
    enable_tfidf: bool = True,
    enable_semantic: bool = True,
    enable_authors: bool = True,
    enable_numerics: bool = True,
    enable_image: bool = True,
    tfidf_params: dict | None = None,
    semantic_params: dict | None = None,
    image_params: dict | None = None,
) -> Reps:
    df = ensure_clean_df(df)
    artifacts_root = Path(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    tfidf_params = tfidf_params or {}
    semantic_params = semantic_params or {}
    image_params = image_params or {}

    X_tfidf = tfidf_vec = None
    if enable_tfidf:
        corpus, blank_ratio = lexical.build_tfidf_corpus(
            df,
            blank_non_english=_safe(tfidf_params, "blank_non_english", True),
            title_boost=_safe(tfidf_params, "title_boost", 2),
        )
        tfidf_vec, X_tfidf = lexical.fit_tfidf(
            corpus,
            run_tag=run_tag_tfidf,
            df=df,
            ngram_range=_safe(tfidf_params, "ngram_range", (1, 2)),
            min_df=_safe(tfidf_params, "min_df", 5),
            max_df=_safe(tfidf_params, "max_df", 0.90),
            max_features=_safe(tfidf_params, "max_features", 120_000),
            sublinear_tf=_safe(tfidf_params, "sublinear_tf", True),
            norm=_safe(tfidf_params, "norm", "l2"),
            artifacts_root=artifacts_root / "tfidf",
            force_recompute=_safe(tfidf_params, "force_recompute", False),
        )
        log.info("TF-IDF ready: shape=%s nnz=%s blank_nonEN=%.1f%%", getattr(X_tfidf, "shape", None), getattr(X_tfidf, "nnz", 0), 100*blank_ratio)

    authors_mlb = authors_iaf = X_auth = None
    if enable_authors:
        authors_mlb, authors_iaf = metadata.fit_authors(df, use_canonical=True, artifacts_dir=artifacts_root / "metadata")
        X_auth = metadata.transform_authors(df, authors_mlb, authors_iaf, use_canonical=True)
        X_auth = normalize(X_auth, norm="l2", copy=False)
        log.info("Authors ready: shape=%s nnz=%s", getattr(X_auth, "shape", None), getattr(X_auth, "nnz", 0))

    num_feature_names: list[str] = []
    X_num_raw = X_num = None
    if enable_numerics:
        num_cfg = metadata.fit_numeric_scalers(df, artifacts_dir=artifacts_root / "metadata")
        X_num_raw = metadata.transform_numerics(df, num_cfg)
        X_num = normalize(X_num_raw, norm="l2")
        num_feature_names = list(num_cfg["numeric_feature_names"])
        log.info("Numerics ready: shape=%s", getattr(X_num, "shape", None))

    Z_sem = None
    if enable_semantic:
        try:
            Z_sem = semantic.fit_semantic(
                df,
                run_tag=run_tag_sem,
                artifacts_root=artifacts_root / "semantic",
                force_recompute=_safe(semantic_params, "force_recompute", False),
                device=_safe(semantic_params, "device", None),
                normalize=_safe(semantic_params, "normalize", True),
            )
            log.info("Semantic ready: shape=%s", getattr(Z_sem, "shape", None))
        except Exception as e:
            log.warning("Semantic skipped: %s", e)
            Z_sem = None

    Z_img = img_mask = None
    if enable_image:
        try:
            Z_img, img_mask = image.fit_image_embeddings(
                df,
                run_tag=run_tag_img,
                artifacts_root=artifacts_root / "image",
                covers_cache_dir=artifacts_root / "image" / "covers",
                device=_safe(image_params, "device", None),
                batch_size=_safe(image_params, "batch_size", 32),
                force_recompute=_safe(image_params, "force_recompute", False),
            )
            log.info("Image ready: shape=%s coverage=%.1f%%", getattr(Z_img, "shape", None), 100*float(img_mask.mean()))
        except Exception as e:
            log.warning("Image skipped: %s", e)
            Z_img, img_mask = None, None

    man = {
        "enable": {
            "tfidf": enable_tfidf, "semantic": enable_semantic, "authors": enable_authors,
            "numerics": enable_numerics, "image": enable_image
        },
        "shapes": {
            "tfidf": getattr(X_tfidf, "shape", None),
            "semantic": None if Z_sem is None else list(Z_sem.shape),
            "authors": getattr(X_auth, "shape", None),
            "numerics": None if X_num is None else list(X_num.shape),
            "image": None if Z_img is None else list(Z_img.shape),
        },
        "params": {
            "tfidf": {**{"run_tag": run_tag_tfidf}, **tfidf_params},
            "semantic": {**{"run_tag": run_tag_sem}, **semantic_params},
            "image": {**{"run_tag": run_tag_img}, **image_params},
        },
    }
    (artifacts_root / "manifest_all.json").write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    return Reps(
        X_tfidf=X_tfidf,
        tfidf_vec=tfidf_vec,
        Z_sem=Z_sem,
        X_auth=X_auth,
        authors_mlb=authors_mlb,
        authors_iaf=authors_iaf,
        X_num_raw=X_num_raw,
        X_num=X_num,
        num_feature_names=num_feature_names,
        Z_img=Z_img,
        img_mask=img_mask,
        manifest=man,
    )
