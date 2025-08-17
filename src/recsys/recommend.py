from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from scipy import sparse
from .fusion import rank_percentile, renorm_weights, blend_ranked, mmr_rerank_generic
from src.recsys.utils import cosine_sim_dense, cosine_sim_csr, enforce_caps
import logging

log = logging.getLogger(__name__)

def _get_index_by_book_id(df: pd.DataFrame, book_id: int) -> int:
    if "book_id" in df.columns:
        rows = df.index[df["book_id"] == int(book_id)].tolist()
        if not rows:
            raise KeyError(f"book_id {book_id} not found")
        return int(rows[0])
    if int(book_id) in df.index:
        return int(book_id)
    raise KeyError("book_id not found")

def _sim_track_for_idx(track, idx: int) -> np.ndarray:
    if track is None:
        return np.zeros(0, dtype=np.float32)
    if sparse.issparse(track):
        s = cosine_sim_csr(track[idx], track).toarray().ravel().astype(np.float32)
    else:
        s = cosine_sim_dense(track[idx:idx + 1], track).ravel().astype(np.float32)
    s[idx] = -np.inf
    return s

def _global_image_ok(img_mask: Optional[np.ndarray], seed_idx: int, min_cov: float) -> bool:
    if img_mask is None or img_mask.size == 0:
        return False
    if not bool(img_mask[seed_idx]):
        return False
    return float(np.mean(img_mask.astype(bool))) >= float(min_cov)

def _sim_callable(space: str, reps) -> Optional[Callable[[int, int], float]]:
    spaces = [s.strip() for s in str(space).split("|") if s.strip()]
    sims: List[Callable[[int, int], float]] = []

    for sp in spaces:
        if sp == "semantic" and getattr(reps, "Z_sem", None) is not None:
            Z = reps.Z_sem
            sims.append(lambda i, j, Z=Z: float(np.dot(Z[i], Z[j])))
        elif sp == "numerics" and getattr(reps, "X_num", None) is not None:
            X = reps.X_num
            sims.append(lambda i, j, X=X: float(np.dot(X[i], X[j])))
        elif sp == "image" and getattr(reps, "Z_img", None) is not None:
            Z = reps.Z_img
            sims.append(lambda i, j, Z=Z: float(np.dot(Z[i], Z[j])))
        elif sp == "tfidf" and getattr(reps, "X_tfidf", None) is not None:
            X = reps.X_tfidf
            sims.append(lambda i, j, X=X: float(X[i].multiply(X[j]).sum()) if sparse.issparse(X) else float(np.dot(X[i], X[j])))
        elif sp == "authors" and getattr(reps, "X_auth", None) is not None:
            X = reps.X_auth
            sims.append(lambda i, j, X=X: float(X[i].multiply(X[j]).sum()) if sparse.issparse(X) else float(np.dot(X[i], X[j])))

    if not sims:
        return None
    if len(sims) == 1:
        return sims[0]

    def sim_max(i: int, j: int) -> float:
        return max(fn(i, j) for fn in sims)

    return sim_max


def recommend_core(df: pd.DataFrame, reps, seed_idx: int, k: int, weights: Dict[str, float], lam: float, use_mmr: bool, mmr_space: str, include_year_bias: bool, year_bias_weight: float = 0.0, year_tiebreak: bool = False, caps: Optional[Tuple[int, int]] = None, image_min_coverage: float = 0.3) -> Tuple[List[int], Dict[str, np.ndarray]]:
    sims: Dict[str, np.ndarray] = {}
    if getattr(reps, "X_tfidf", None) is not None:
        sims["tfidf"] = _sim_track_for_idx(reps.X_tfidf, seed_idx)
    if getattr(reps, "Z_sem", None) is not None and isinstance(reps.Z_sem, np.ndarray) and reps.Z_sem.size:
        sims["semantic"] = _sim_track_for_idx(reps.Z_sem, seed_idx)
    if getattr(reps, "X_auth", None) is not None:
        sims["authors"] = _sim_track_for_idx(reps.X_auth, seed_idx)
    if getattr(reps, "X_num", None) is not None:
        sims["numerics"] = _sim_track_for_idx(reps.X_num, seed_idx)
    if getattr(reps, "Z_img", None) is not None and getattr(reps, "img_mask", None) is not None:
        if _global_image_ok(reps.img_mask, seed_idx, image_min_coverage):
            sims["image"] = _sim_track_for_idx(reps.Z_img, seed_idx)
    if not sims:
        log.error("recommend_core no_modalities seed=%d", seed_idx)
        return [], {}
    weights_use = renorm_weights(weights, list(sims.keys()))
    pct = {}
    for kmod, s in sims.items():
        if kmod == "image" and getattr(reps, "img_mask", None) is not None:
            mask = np.isfinite(s) & reps.img_mask.astype(bool)
        else:
            mask = np.isfinite(s)
        pct[kmod] = rank_percentile(s, mask)
    fused = blend_ranked(pct, weights_use)
    if include_year_bias and year_bias_weight > 0.0:
        df_num_raw = pd.DataFrame(reps.X_num_raw, columns=reps.num_feature_names, index=df.index)
        seed_mod = bool(pd.notna(df_num_raw.loc[seed_idx, "modern_flag"]) and df_num_raw.loc[seed_idx, "modern_flag"])
        seed_pre = bool(pd.notna(df_num_raw.loc[seed_idx, "pre_1900_flag"]) and df_num_raw.loc[seed_idx, "pre_1900_flag"])
        mod_series = df_num_raw["modern_flag"].fillna(False).astype(bool)
        pre_series = df_num_raw["pre_1900_flag"].fillna(False).astype(bool)
        match = (mod_series == seed_mod) & (pre_series == seed_pre)
        fused = fused + year_bias_weight * match.to_numpy(dtype=np.float32)
    order = np.argsort(-fused)
    order = [int(i) for i in order if np.isfinite(fused[i])]
    if use_mmr:
        sim_ij = _sim_callable(mmr_space, reps)
        if sim_ij is not None:
            picked = mmr_rerank_generic(fused, k=k, lam=lam, sim_ij=sim_ij)
        else:
            picked = order[:k]
    else:
        picked = order[:k]
    if year_tiebreak and len(picked) > 1:
        y0 = pd.to_numeric(df.loc[seed_idx, "original_publication_year"], errors="coerce")
        def yd(i):
            yi = pd.to_numeric(df.loc[i, "original_publication_year"], errors="coerce")
            if pd.isna(y0) or pd.isna(yi):
                return np.inf
            return abs(float(yi) - float(y0))
        picked = sorted(picked, key=yd)
    if caps is not None:
        a_cap, d_cap = caps
        picked = enforce_caps(picked, df, max_per_author=int(a_cap), max_per_desc=int(d_cap))[:k]
    return picked[:k], pct

def recommend_by_book_id(df: pd.DataFrame, reps, book_id: int, k: int = 5, weights: Dict[str, float] | None = None, lam: float = 0.95, use_mmr: bool = True, mmr_space: str = "semantic", include_year_bias: bool = False, year_bias_weight: float = 0.0, year_tiebreak: bool = False, apply_caps: bool = True, image_min_coverage: float = 0.3) -> Dict:
    if weights is None:
        weights = {"tfidf": 0.3, "semantic": 0.4, "authors": 0.2, "numerics": 0.1, "image": 0.0}
    idx = _get_index_by_book_id(df, int(book_id))
    caps = (3, 1) if apply_caps else None
    picked, pct = recommend_core(df, reps, idx, k=k, weights=weights, lam=lam, use_mmr=use_mmr, mmr_space=mmr_space, include_year_bias=include_year_bias, year_bias_weight=year_bias_weight, year_tiebreak=year_tiebreak, caps=caps, image_min_coverage=image_min_coverage)
    rows = [{"book_id": int(df.loc[i, "book_id"]), "row_index": int(i)} for i in picked]
    log.info("recommend_by_book_id seed_book_id=%d k=%d use_mmr=%s caps=%s", int(book_id), k, use_mmr, bool(apply_caps))
    return {"seed_book_id": int(book_id), "picked": rows, "modality_percentiles": {k: pct[k] for k in pct}}
