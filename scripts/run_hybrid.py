import os, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.clean.pipeline import clean_books_dataset
from src.features_representation.pipeline import fit_or_load_all
from src.recsys.genres import load_cache, save_cache, build_igf_weights
# from src.recsys.genres_api import hydrate_cache_google
from src.recsys.genres_api_openlib import hydrate_cache_openlib
from src.recsys.grid_search import weight_grid, evaluate_weights
from src.recsys.recommend import recommend_by_book_id
from src.recsys.utils import _to_serializable

try:
    from src.config import GOOGLE_BOOKS_API_KEY
except Exception:
    GOOGLE_BOOKS_API_KEY = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("hybrid-runner")

ARTIFACTS = Path("artifacts")
RUN_TAG = "hybrid_v1_rankpct"
RECSYS_DIR = ARTIFACTS / "recsys" / RUN_TAG
RECSYS_DIR.mkdir(parents=True, exist_ok=True)
GENRE_CACHE_FP = RECSYS_DIR / "genre_cache.json"

def ensure_genre_cache(df: pd.DataFrame, max_updates: int = 100) -> dict:
    cache = load_cache(GENRE_CACHE_FP)
    if not cache:
        log.info("ensure_genre_cache(miss) path=%s", GENRE_CACHE_FP)
        cache = hydrate_cache_openlib(df, GENRE_CACHE_FP, max_updates=max_updates)
    cov = np.mean([1 if str(int(b)) in cache else 0 for b in df["book_id"].tolist()])
    log.info("ensure_genre_cache(openlib) coverage=%.3f size=%d", cov, len(cache))
    sample_keys = list(cache.keys())[:5]
    log.info("genre sample keys=%s", sample_keys)
    for k in sample_keys:
        log.info("subjects for %s -> %s", k, cache[k][:5])
    return cache


def eval_and_select_weights(df_clean: pd.DataFrame, reps, genre_cache: dict, use_mmr: bool = True, lam: float = 0.95, mmr_space: str = "tfidf|authors", max_test_size: int = 100, max_test_size_ratio: float = 0.5) -> dict:
    seeds_all = df_clean.index.tolist()
    test_size = min(int(len(seeds_all) * max_test_size_ratio), max_test_size)
    if len(seeds_all) > test_size:
        rng = np.random.default_rng(42)
        seeds = rng.choice(seeds_all, size=test_size, replace=False).tolist()
    else:
        seeds = seeds_all
    igf_combo = build_igf_weights(genre_cache, mode="combo")
    grids = weight_grid()
    rows = []
    for w in grids:
        m = evaluate_weights(df_clean, reps, seeds=seeds, weights=w, genre_cache=genre_cache, igf=igf_combo, k_eval=10, lam=lam, use_mmr=use_mmr, mmr_space=mmr_space)
        rows.append({"weights": w} | m)
    res_df = pd.DataFrame([{"tfidf": r["weights"]["tfidf"], "semantic": r["weights"]["semantic"], "authors": r["weights"]["authors"], "numerics": r["weights"]["numerics"], "image": r["weights"]["image"], "Genre@K": r["Genre@K"], "YearFlag@K": r["YearFlag@K"], "EvalScore": r["EvalScore"]} for r in rows]).sort_values("EvalScore", ascending=False)
    res_fp = RECSYS_DIR / "grid_eval_summary.csv"
    res_df.to_csv(res_fp, index=False)
    best = res_df.iloc[0].to_dict()
    best_w = {"tfidf": float(best["tfidf"]), "semantic": float(best["semantic"]), "authors": float(best["authors"]), "numerics": float(best["numerics"]), "image": float(best["image"])}
    with open(RECSYS_DIR / "weights_selected.json", "w", encoding="utf-8") as f:
        json.dump(best_w, f, ensure_ascii=False, indent=2)
    log.info("eval_and_select_weights selected=%s path=%s", best_w, res_fp)
    return best_w

def serve_final(df_clean: pd.DataFrame, reps, seed_book_id: int, weights: dict, year_bias_weight: float, use_mmr: bool, mmr_space: str) -> dict:
    rec = recommend_by_book_id(
        df_clean,
        reps,
        book_id=seed_book_id,
        k=5,
        weights=weights,
        lam=0.95,
        use_mmr=use_mmr,
        mmr_space=mmr_space,
        include_year_bias=True,
        year_bias_weight=year_bias_weight,
        year_tiebreak=True,
        apply_caps=True,
        image_min_coverage=0.3
    )
    log.info("serve_final seed_book_id=%d use_mmr=%s mmr_space=%s year_alpha=%.3f", seed_book_id, use_mmr, mmr_space, year_bias_weight)
    return rec

def main():
    mmr_space = "tfidf|authors"
    df = pd.read_csv("data/raw/book.csv")
    df_clean = clean_books_dataset(df, drop_language_col=True)
    reps = fit_or_load_all(df_clean, artifacts_root=ARTIFACTS / "eda2", enable_image=True, run_tag_img="dinov2_vits14")
    genre_cache = ensure_genre_cache(df_clean, max_updates=100)
    best_w = eval_and_select_weights(df_clean, reps, genre_cache, use_mmr=True, lam=0.95, mmr_space=mmr_space)
    seed_idx = 0
    seed_book_id = int(df_clean.loc[seed_idx, "book_id"])
    post_year_alpha = 0.02
    mmr_on = False
    rec = serve_final(df_clean, reps, seed_book_id, weights=best_w, year_bias_weight=post_year_alpha, use_mmr=mmr_on, mmr_space=mmr_space)
    out_csv = RECSYS_DIR / f"top5_final_{seed_book_id}.csv"
    ids = [r["book_id"] for r in rec["picked"]]
    pd.DataFrame({
        "rank": list(range(1, len(ids) + 1)),
        "book_id": ids,
        "title": [df_clean.loc[df_clean.index[df_clean["book_id"] == bid][0], "title"] for bid in ids],
        "authors": [df_clean.loc[df_clean.index[df_clean["book_id"] == bid][0], "authors"] for bid in ids],
        "year": [df_clean.loc[df_clean.index[df_clean["book_id"] == bid][0], "original_publication_year"] for bid in ids],
        "avg_rating": [df_clean.loc[df_clean.index[df_clean["book_id"] == bid][0], "average_rating"] for bid in ids],
    }).to_csv(out_csv, index=False)
    with open(RECSYS_DIR / f"top5_final_{seed_book_id}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2, default=_to_serializable)
    log.info("outputs csv=%s json=%s", out_csv, RECSYS_DIR / f"top5_final_{seed_book_id}.json")

if __name__ == "__main__":
    main()
