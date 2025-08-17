import numpy as np
import pandas as pd
from collections import Counter
from src.recsys.utils import hhi, cosine_sim_dense
import logging
log = logging.getLogger(__name__)

def uniqueness_from_agreement(agreement_df, anchor="semantic"):
    if isinstance(agreement_df, dict):
        rows = []
        for k, v in agreement_df.items():
            rows.append({"pair": k, "jaccard@10": v[0], "kendall_tau@10": v[1]})
        df = pd.DataFrame(rows)
    else:
        df = agreement_df.copy()
    out = {}
    for r in df.itertuples():
        if isinstance(r.pair, str) and r.pair.startswith(f"{anchor}_"):
            mod = r.pair.split("_", 1)[1]
            out[mod] = 1.0 - float(r._2)
    return out

def weight_scorecard(baseline, stability, separation, uniqueness, include_image=True, image_cap=0.1, other_cap=0.6, w_stab=0.4, w_sep=0.4, w_uniq=0.1):
    tracks = sorted(set(list(baseline.keys()) + list(stability.keys()) + list(separation.keys()) + list(uniqueness.keys())))
    raw = {}
    for t in tracks:
        st = float(stability.get(t, 0.0))
        se = float(separation.get(t, 0.0))
        un = float(uniqueness.get(t, 0.0)) if t != "semantic" else 0.0
        s = w_stab * st + w_sep * se + w_uniq * un
        raw[t] = max(0.0, s)
    caps = {t: (image_cap if t == "image" else other_cap) for t in tracks}
    if not include_image and "image" in caps:
        caps["image"] = 0.0
        raw["image"] = 0.0
    capped = {t: min(caps[t], raw[t]) for t in tracks}
    tot = sum(capped.values())
    rec = {t: (capped[t] / tot if tot > 0 else 0.0) for t in tracks}
    rows = []
    for t in tracks:
        log.info(   
            "Track: %s | Baseline: %.3f | Recommended: %.3f | Stability: %.3f | Separation: %.3f | Uniqueness: %.3f",
            t, baseline.get(t, 0.0), rec[t], stability.get(t, 0.0), separation.get(t, 0.0), uniqueness.get(t, 0.0) if t != "semantic" else 0.0
        )
        rows.append({"track": t, "baseline": float(baseline.get(t, 0.0)), "recommended": float(rec.get(t, 0.0)), "stability": float(stability.get(t, 0.0)), "separation": float(separation.get(t, 0.0)), "uniqueness": float(uniqueness.get(t, 0.0) if t != "semantic" else 0.0), "cap": float(caps[t])})
    df = pd.DataFrame(rows).sort_values("track")
    return rec, df

def deltas_over_threshold(rec_df, thr=0.05):
    d = rec_df.copy()
    d["delta"] = d["recommended"] - d["baseline"]
    return d.loc[d["delta"].abs() >= thr].sort_values("delta", ascending=False)

def diversity_check(modality_matrix, df, k=10):
    sims = cosine_sim_dense(modality_matrix, modality_matrix)
    np.fill_diagonal(sims, -np.inf)
    top_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]

    hhi_scores = []
    year_spreads = []

    for i in range(top_idx.shape[0]):
        authors = df.iloc[top_idx[i]]["authors"].tolist()
        years = df.iloc[top_idx[i]]["original_publication_year"].dropna().tolist()
        hhi_scores.append(hhi(authors))
        if years:
            year_spreads.append(max(years) - min(years))
        else:
            year_spreads.append(0)

    return np.array(hhi_scores), np.array(year_spreads)