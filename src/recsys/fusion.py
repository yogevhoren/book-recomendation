from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Callable
import logging

log = logging.getLogger(__name__)

def rank_percentile(scores: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    s = scores.astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(s)
    idx = np.where(valid_mask)[0]
    out = np.zeros_like(s, dtype=np.float32)
    if idx.size == 0:
        log.warning("rank_percentile no_valid idx=0")
        return out
    vals = s[idx]
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(order.size)
    pct = (ranks.astype(np.float32) + 1.0) / (order.size + 1.0)
    out[idx] = pct
    log.debug("rank_percentile done n=%d valid=%d", s.size, idx.size)
    return out

def renorm_weights(weights: Dict[str, float], available: List[str]) -> Dict[str, float]:
    w = {k: float(weights.get(k, 0.0)) for k in available}
    log.info("renorm_weights initial=%s available=%s", w, available)
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        w = {k: 1.0 / len(available) for k in available}
        log.warning("renorm_weights zero_sum using_uniform available=%s", available)
        return w
    w = {k: max(0.0, v) / s for k, v in w.items()}
    log.info("renorm_weights final=%s", w)
    return w

def blend_ranked(mod2pct: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    n = len(next(iter(mod2pct.values())))
    fused = np.zeros(n, dtype=np.float32)
    for k, v in mod2pct.items():
        fused += v * float(weights.get(k, 0.0))
    log.info("blend_ranked mods=%s n=%d", list(mod2pct.keys()), n)
    return fused

def mmr_rerank_generic(fused: np.ndarray, k: int, lam: float, sim_ij: Callable[[int, int], float]) -> List[int]:
    n = fused.shape[0]
    if n == 0 or k <= 0:
        log.warning("mmr_rerank_generic empty_input n=%d k=%d", n, k)
        return []
    order = np.argsort(-fused)
    picked: List[int] = []
    chosen = np.zeros(n, dtype=bool)
    for i0 in order:
        picked.append(int(i0))
        chosen[i0] = True
        break
    while len(picked) < min(k, n):
        best_idx = -1
        best_score = -np.inf
        for i in order:
            if chosen[i]:
                continue
            rel = fused[i]
            div = 0.0
            for j in picked:
                s = sim_ij(int(i), int(j))
                div = max(div, 1.0 - float(s))
            score = lam * rel - (1.0 - lam) * div
            if score > best_score:
                best_score = score
                best_idx = int(i)
        if best_idx < 0:
            break
        picked.append(best_idx)
        chosen[best_idx] = True
    log.info("mmr_rerank_generic k=%d lam=%.3f picked=%s", k, lam, picked)
    return picked
