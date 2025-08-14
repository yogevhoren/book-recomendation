from __future__ import annotations
import logging, re
from typing import List, Dict, Any, Set
import pandas as pd

log = logging.getLogger(__name__)

def norm_title_for_dedup(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def author_list(authors: str) -> List[str]:
    return [a.strip().lower() for a in str(authors).split(",") if a.strip()]

def consolidate_by_title_author_overlap(df: pd.DataFrame, log_examples: int = 3) -> pd.DataFrame:
    df = df.copy()
    start = len(df)
    df["title_norm"] = df["title"].map(norm_title_for_dedup)
    if "author_list" not in df.columns:
        df["author_list"] = df["authors"].map(author_list)

    out_rows: List[Dict[str, Any]] = []
    merged_summaries: List[Dict[str, Any]] = []

    for tnorm, g in df.groupby("title_norm", sort=False):
        idxs = list(g.index)
        if len(idxs) == 1:
            out_rows.append(df.loc[idxs[0]].to_dict())
            continue

        auth_sets: Dict[int, Set[str]] = {i: set(df.at[i, "author_list"]) for i in idxs}
        adj: Dict[int, Set[int]] = {i: set() for i in idxs}
        for i in idxs:
            for j in idxs:
                if i >= j:
                    continue
                if auth_sets[i] & auth_sets[j]:
                    adj[i].add(j); adj[j].add(i)

        seen, comps = set(), []
        for i in idxs:
            if i in seen:
                continue
            stack, comp = [i], set()
            while stack:
                v = stack.pop()
                if v in seen:
                    continue
                seen.add(v); comp.add(v)
                stack.extend(adj[v] - seen)
            comps.append(comp)

        for comp in comps:
            comp = list(comp)
            sub = df.loc[comp]
            if len(comp) == 1:
                out_rows.append(sub.iloc[0].to_dict())
                continue

            if "original_publication_year" in sub:
                rep_idx = sub["original_publication_year"].idxmin()
                if pd.isna(df.at[rep_idx, "original_publication_year"]):
                    rep_idx = comp[0]
            else:
                rep_idx = comp[0]

            rep = df.loc[rep_idx].to_dict()
            merged_authors = sorted({a for lst in sub["author_list"] for a in lst})
            rep["authors"] = ", ".join(merged_authors)
            rep["author_list"] = merged_authors
            if "average_rating" in sub:
                rep["average_rating"] = float(sub["average_rating"].mean())

            out_rows.append({str(k): v for k, v in rep.items()})
            merged_summaries.append({
                "title_norm": tnorm,
                "rows": int(len(sub)),
                "years": sorted(sub["original_publication_year"].dropna().unique().tolist()),
            })

    out = pd.DataFrame(out_rows).drop(columns=["title_norm"], errors="ignore").reset_index(drop=True)
    removed = start - len(out)
    if removed:
        log.info(f"Deduplicated by title + shared author. Removed {removed} rows ({removed/start:.2%}).")
        for eg in merged_summaries[:log_examples]:
            log.info(f"  · '{eg['title_norm']}' merged {eg['rows']} rows; years={eg['years']}")
    else:
        log.info("No duplicates found by title + shared author.")
    return out
