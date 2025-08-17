from __future__ import annotations
import numpy as np
from typing import Dict, List
import logging

log = logging.getLogger(__name__)

def per_item_contrib(mod2pct: Dict[str, np.ndarray], weights: Dict[str, float], picked: List[int]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for i in picked:
        row = {}
        for k, v in mod2pct.items():
            row[k] = float(v[i] * weights.get(k, 0.0))
        out.append(row)
    log.debug("per_item_contrib k=%d mods=%s", len(picked), list(mod2pct.keys()))
    return out

def numeric_feature_contrib(q: np.ndarray, X_num: np.ndarray, picked: List[int], feature_names: List[str]) -> List[Dict[str, float]]:
    res: List[Dict[str, float]] = []
    for i in picked:
        v = X_num[i]
        c = q * v
        res.append({feature_names[j]: float(c[j]) for j in range(len(feature_names))})
    log.debug("numeric_feature_contrib k=%d d=%d", len(picked), len(feature_names))
    return res
