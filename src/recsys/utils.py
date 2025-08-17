import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import pairwise_distances
from umap import UMAP
from typing import List, Optional
import logging
log = logging.getLogger(__name__)


def cosine_sim_dense(A, B):
    return 1 - pairwise_distances(A, B, metric="cosine")

def cosine_sim_csr(A, B=None):
    X = A
    Y = A if B is None else B
    return X @ Y.T

def topk_sim(sim, k):
    S = sim.toarray() if sparse.issparse(sim) else sim
    k = min(k, S.shape[1] - 1)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    part = np.take_along_axis(S, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)

def topk_indices_from_track(track, k):
    if sparse.issparse(track):
        S = cosine_sim_csr(track).toarray()  
    else:
        S = cosine_sim_dense(track, track)
    np.fill_diagonal(S, -1)
    idx, _ = topk_sim(S, k)
    return idx

def jaccard_at_k(ranks_a, ranks_b):
    k = ranks_a.shape[1]
    out = []
    for a, b in zip(ranks_a, ranks_b):
        out.append(len(set(a[:k]).intersection(b[:k])) / k)
    return np.array(out)

def kendall_tau_approx(ranks_a, ranks_b):
    k = ranks_a.shape[1]
    out = []
    for a, b in zip(ranks_a, ranks_b):
        pa = {v: i for i, v in enumerate(a[:k])}
        pb = {v: i for i, v in enumerate(b[:k])}
        inter = [v for v in a[:k] if v in pb]
        if len(inter) < 2:
            out.append(0.0)
            continue
        seq = [pb[v] for v in inter]
        inv = 0
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if seq[i] > seq[j]:
                    inv += 1
        n = len(seq)
        d = n * (n - 1) / 2
        out.append(1 - 2 * inv / d if d > 0 else 0.0)
    return np.array(out)

def hhi(items):
    if len(items) == 0:
        return 0.0
    _, cnts = np.unique(items, return_counts=True)
    p = cnts / np.sum(cnts)
    return float(np.sum(p * p))

def sample_negatives(n, k, exclude):
    res = []
    for i in range(n):
        size = min(max(1, n - 1), k * 3)
        cand = np.random.choice(n, size=size, replace=False)
        cand = [c for c in cand if c != i and c not in exclude.get(i, [])]
        res.append(np.array(cand[:k]))
    return res

def pos_neg_index_sets(df, k=30):
    n = len(df)
    main_author = df["author_list"].apply(lambda xs: xs[0].strip().lower() if isinstance(xs, list) and len(xs) > 0 else "").tolist()
    author_index = {}
    for i, a in enumerate(main_author):
        if a:
            author_index.setdefault(a, []).append(i)
    pos_by_desc = {}
    for _, g in df.groupby("desc_hash"):
        idxs = g.index.tolist()
        if len(idxs) >= 2:
            for i in idxs:
                pos_by_desc.setdefault(i, set()).update(j for j in idxs if j != i)
    pos_by_author = {}
    for _, idxs in author_index.items():
        if len(idxs) >= 2:
            for i in idxs:
                pos_by_author.setdefault(i, set()).update(j for j in idxs if j != i)
    pos = [[] for _ in range(n)]
    for i in range(n):
        s = set()
        if i in pos_by_desc:
            s.update(pos_by_desc[i])
        if i in pos_by_author:
            s.update(pos_by_author[i])
        pos[i] = np.array(list(s))[:k] if len(s) > 0 else np.array([], dtype=int)
    neg = sample_negatives(n, k, exclude={i: pos[i] for i in range(n)})
    return pos, neg

def compute_norm_outlier_share(X, mad_thresh=5):
    V = X.toarray() if sparse.issparse(X) else X
    n = np.linalg.norm(V, axis=1)
    med = np.median(n)
    mad = np.median(np.abs(n - med))
    return float(np.mean(np.abs(n - med) / (mad + 1e-9) > mad_thresh))

def umap_embed(X, n_neighbors=15, random_state=42):
    V = X.toarray() if sparse.issparse(X) else X
    return UMAP(n_neighbors=n_neighbors, n_components=2, random_state=random_state, metric="cosine", min_dist=0.1).fit_transform(V)

def resample_stability(track, k=100, trials=3, frac=0.5, seed=42):
    rng = np.random.default_rng(seed)
    base = topk_indices_from_track(track, k)
    scores = []
    N = track.shape[0]
    for _ in range(trials):
        mask = rng.random(N) < frac
        sub = track[mask]
        sub_idx = topk_indices_from_track(sub, k)
        map_back = np.where(mask)[0]
        reb = np.full_like(base, -1)
        for i in range(N):
            if mask[i]:
                sidx = np.where(map_back == i)[0]
                if len(sidx) == 1:
                    reb[i, :] = map_back[sub_idx[sidx[0]]]
        ok = reb[:, 0] >= 0
        scores.append(jaccard_at_k(base[ok, :10], reb[ok, :10]).mean() if ok.any() else 0.0)
    return float(np.mean(scores))

def tfidf_top_terms(vectorizer, text, top=15):
    v = vectorizer.transform([text]).toarray()[0]
    order = np.argsort(-v)[:top]
    vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    return [vocab[i] for i in order if v[i] > 0]

def nearest_table(track, df, seed_idx, top=10):
    if sparse.issparse(track):
        s = cosine_sim_csr(track[seed_idx], track).toarray().ravel()
    else:
        s = cosine_sim_dense(track[seed_idx:seed_idx + 1], track).ravel()
    s[seed_idx] = -1
    order = np.argsort(-s)[:top]
    base_auth = set(df.loc[seed_idx, "author_list"] if isinstance(df.loc[seed_idx, "author_list"], list) else [])
    rows = []
    y0 = pd.to_numeric(df.loc[seed_idx, "original_publication_year"], errors="coerce")
    r0 = pd.to_numeric(df.loc[seed_idx, "average_rating"], errors="coerce")
    for j in order:
        aj = set(df.loc[j, "author_list"] if isinstance(df.loc[j, "author_list"], list) else [])
        y = pd.to_numeric(df.loc[j, "original_publication_year"], errors="coerce")
        r = pd.to_numeric(df.loc[j, "average_rating"], errors="coerce")
        rows.append({
            "book_id": int(df.loc[j, "book_id"]),
            "title": df.loc[j, "title"],
            "cosine": float(s[j]),
            "shared_author": bool(len(base_auth.intersection(aj)) > 0),
            "d_year": float((y - y0) if pd.notna(y) and pd.notna(y0) else np.nan),
            "d_rating": float((r - r0) if pd.notna(r) and pd.notna(r0) else np.nan)
        })
    return pd.DataFrame(rows)

def enforce_caps(ranked_ids, df, max_per_author=3, max_per_desc=2):
    kept = []
    author_counts = {}
    desc_counts = {}
    for bid in ranked_ids:
        row = df.loc[bid]
        authors = row["author_list"] if isinstance(row["author_list"], list) else []
        desc_hash = row["desc_hash"]
        if authors:
            main_author = authors[0].strip().lower()
            if author_counts.get(main_author, 0) >= max_per_author:
                continue
        else:
            main_author = None
        if desc_counts.get(desc_hash, 0) >= max_per_desc:
            continue
        kept.append(bid)
        if main_author:
            author_counts[main_author] = author_counts.get(main_author, 0) + 1
        desc_counts[desc_hash] = desc_counts.get(desc_hash, 0) + 1
    return kept

def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def pick_seeds_with_outliers(
    df: pd.DataFrame,
    reps,
    n_random: int = 300,
    rng: Optional[np.random.Generator] = None,
    include_missing_desc: bool = True,
    include_pre1900: bool = True,
    include_modern: bool = True,
    include_no_image: bool = True,
    include_popular: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    if rng is None:
        rng = np.random.default_rng(42)

    idx_all = df.index.to_numpy()
    if idx_all.size == 0:
        return []

    n_random = int(min(max(n_random, 0), idx_all.size))
    outliers: List[int] = []

    df_num_raw = None
    if getattr(reps, "X_num_raw", None) is not None and getattr(reps, "num_feature_names", None) is not None:
        try:
            df_num_raw = pd.DataFrame(reps.X_num_raw, columns=reps.num_feature_names, index=df.index)
            log.info("df_num_raw good. pick_seeds_with_outliers: num_raw features loaded with %d rows", len(df_num_raw))
        except Exception:
            df_num_raw = None

    if include_missing_desc and "description" in df.columns:
        mdesc = df.index[(df["description"].isna()) | (df["description"].astype(str).str.len() == 0)].tolist()
        if mdesc:
            outliers.append(int(mdesc[0]))

    if include_pre1900 and df_num_raw is not None and "pre_1900_flag" in df_num_raw.columns:
        pre_list = df_num_raw["pre_1900_flag"].fillna(False).astype(bool)
        if pre_list.any():
            outliers.append(int(pre_list[0]))

    if include_modern and df_num_raw is not None and "modern_flag" in df_num_raw.columns:
        mod_list = df_num_raw["modern_flag"].fillna(False).astype(bool)
        if mod_list.any():
            outliers.append(int(mod_list[0]))

    if include_no_image and getattr(reps, "img_mask", None) is not None:
        try:
            noimg_idx = np.where(~reps.img_mask.astype(bool))[0].tolist()
            if noimg_idx:
                outliers.append(int(df.index[noimg_idx[0]]))
        except Exception:
            pass

    if include_popular and "average_rating" in df.columns:
        df_tmp = df.copy()
        if "description" in df_tmp.columns:
            df_tmp["__desc_len__"] = df_tmp["description"].fillna("").astype(str).str.len()
        else:
            df_tmp["__desc_len__"] = 0
        df_tmp["__ar__"] = pd.to_numeric(df_tmp["average_rating"], errors="coerce").fillna(-1e9)
        df_tmp = df_tmp.sort_values(["__ar__", "__desc_len__"], ascending=[False, False])
        pop_idx = df_tmp.index.tolist()
        if pop_idx:
            outliers.append(int(pop_idx[0]))

    outliers = list(dict.fromkeys([i for i in outliers if i in df.index]))

    pool = np.setdiff1d(idx_all, np.array(outliers, dtype=int), assume_unique=False)
    if pool.size > 0 and n_random > 0:
        random_idxs = rng.choice(pool, size=min(n_random, pool.size), replace=False).tolist()
    else:
        random_idxs = []

    seeds = list(dict.fromkeys(random_idxs + outliers))
    return seeds, random_idxs, outliers
