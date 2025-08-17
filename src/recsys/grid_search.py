from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from .genres import build_igf_weights
from .eval import genre_at_k, yearflag_at_k
from .recommend import recommend_core
import logging

log = logging.getLogger(__name__)

def weight_grid() -> List[Dict[str, float]]:
    grids = []
    for tfidf in [0.2, 0.4]:
        for semantic in [0.3, 0.5]:
            for authors in [0.1, 0.3]:
                for numerics in [0.05, 0.15]:
                    for image in [0.05]:
                        w = {"tfidf": tfidf, "semantic": semantic, "authors": authors, "numerics": numerics, "image": image}
                        s = sum(w.values())
                        if s <= 0:
                            continue
                        grids.append(w)
    log.info("weight_grid size=%d", len(grids))
    return grids

def fine_sweep_around(best: Dict[str, float], step: float = 0.05) -> List[Dict[str, float]]:
    keys = ["tfidf", "semantic", "authors", "numerics", "image"]
    deltas = [-step, 0.0, step]
    out = []
    for dt in deltas:
        for ds in deltas:
            for da in deltas:
                for dn in deltas:
                    for di in deltas:
                        w = {
                            "tfidf": max(0.0, best.get("tfidf", 0.0) + dt),
                            "semantic": max(0.0, best.get("semantic", 0.0) + ds),
                            "authors": max(0.0, best.get("authors", 0.0) + da),
                            "numerics": max(0.0, best.get("numerics", 0.0) + dn),
                            "image": max(0.0, best.get("image", 0.0) + di)
                        }
                        s = sum(w.values())
                        if s == 0:
                            continue
                        w = {k: v / s for k, v in w.items()}
                        out.append(w)
    uniq = []
    seen = set()
    for w in out:
        t = tuple(round(w[k], 3) for k in keys)
        if t not in seen:
            seen.add(t)
            uniq.append(w)
    return uniq


def evaluate_weights(df: pd.DataFrame, reps, seeds: List[int], weights: Dict[str, float], genre_cache: Dict[str, List[str]], igf: Dict, k_eval: int = 10, lam: float = 0.95, use_mmr: bool = True, mmr_space: str = "semantic", genre_weight: float = 0.8, year_flag_weight: float = 0.2) -> Dict[str, float]:
    g_scores = []
    y_scores = []
    for seed_idx in seeds:
        idxs, _ = recommend_core(df, reps, seed_idx, k=k_eval, weights=weights, lam=lam, use_mmr=use_mmr, mmr_space=mmr_space, include_year_bias=False, caps=None, image_min_coverage=0.3)
        g = genre_at_k(seed_idx, idxs, df, genre_cache, igf=igf, mode="combo")
        y = yearflag_at_k(seed_idx, idxs, df, reps)
        g_scores.append(g)
        y_scores.append(y)
    g_mean = float(np.mean(g_scores)) if g_scores else 0.0
    y_mean = float(np.mean(y_scores)) if y_scores else 0.0

    eval_score = genre_weight * g_mean + year_flag_weight * y_mean
    log.info("evaluate_weights tfidf=%.2f sem=%.2f auth=%.2f num=%.2f img=%.2f | Genre@K=%.4f YearFlag@K=%.4f Eval=%.4f",
             weights.get("tfidf", 0.0), weights.get("semantic", 0.0), weights.get("authors", 0.0), weights.get("numerics", 0.0), weights.get("image", 0.0),
             g_mean, y_mean, eval_score)
    return {"Genre@K": g_mean, "YearFlag@K": y_mean, "EvalScore": eval_score}
