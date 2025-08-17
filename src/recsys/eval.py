from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, List, Tuple
from .genres import coarse_map
import logging

log = logging.getLogger(__name__)

def genre_at_k(seed_id: int, topk_ids: Iterable[int], df: pd.DataFrame, genre_cache: Dict[str, List[str]], igf: Dict | None = None, mode: str = "combo") -> float:
    sid = int(df.loc[seed_id, "book_id"])
    seed_raw = genre_cache.get(str(sid), [])
    seed_coarse = coarse_map(seed_raw)
    if not seed_coarse:
        return 0.0
    hits = []
    for i in topk_ids:
        bid = int(df.loc[i, "book_id"])
        raw = genre_cache.get(str(bid), [])
        coarse = coarse_map(raw)
        overlap = seed_coarse.intersection(coarse)
        if not overlap:
            hits.append(0.0)
            continue
        if igf is None:
            hits.append(1.0)
            continue
        if mode == "combo":
            key = tuple(sorted(list(coarse)))
            w = float(igf.get(key, 1.0))
        else:
            w = np.mean([igf.get(g, 1.0) for g in overlap]) if overlap else 0.0
        hits.append(float(w))
    score = float(np.mean(hits)) if hits else 0.0
    log.debug("genre_at_k seed=%d score=%.4f", sid, score)
    return score

def yearflag_at_k(seed_id: int, topk_ids: Iterable[int], df: pd.DataFrame, reps) -> float:
    df_num_raw = pd.DataFrame(reps.X_num_raw, columns=reps.num_feature_names, index=df.index)
    seed_mod_modern_flag = bool(pd.notna(df_num_raw.loc[seed_id, "modern_flag"]) and df_num_raw.loc[seed_id, "modern_flag"])
    seed_pre_1900_flag = bool(pd.notna(df_num_raw.loc[seed_id, "pre_1900_flag"]) and df_num_raw.loc[seed_id, "pre_1900_flag"])
    hits = []
    for i in topk_ids:
        mod_i = bool(pd.notna(df_num_raw.loc[i, "modern_flag"]) and df_num_raw.loc[i, "modern_flag"])
        pre_i = bool(pd.notna(df_num_raw.loc[i, "pre_1900_flag"]) and df_num_raw.loc[i, "pre_1900_flag"])
        hits.append(1.0 if (mod_i == seed_mod_modern_flag and pre_i == seed_pre_1900_flag) else 0.0)
    score = float(np.mean(hits)) if hits else 0.0
    log.debug("yearflag_at_k seed_row=%d seed_mod=%s seed_pre=%s score=%.4f", seed_id, seed_mod_modern_flag, seed_pre_1900_flag, score)
    return score

def rating_band_drift(seed_id: int, topk_ids: Iterable[int], df: pd.DataFrame, band: float = 0.5) -> float:
    r0 = pd.to_numeric(df.loc[seed_id, "average_rating"], errors="coerce")
    diffs = []
    for i in topk_ids:
        ri = pd.to_numeric(df.loc[i, "average_rating"], errors="coerce")
        if pd.isna(ri) or pd.isna(r0):
            continue
        diffs.append(0.0 if abs(float(ri) - float(r0)) <= band else 1.0)
    score = float(np.mean(diffs)) if diffs else 0.0
    log.debug("rating_band_drift seed_row=%d band=%.2f drift=%.4f", seed_id, band, score)
    return score
