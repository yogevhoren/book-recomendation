from __future__ import annotations
import pandas as pd

def exact_title_author_dups(df: pd.DataFrame) -> pd.DataFrame:
    key = ["title", "authors"]
    if not set(key) <= set(df.columns):
        return pd.DataFrame(columns=list(df.columns))
    mask = df.duplicated(subset=key, keep=False)
    return (df.loc[mask, key + ["book_id", "original_publication_year", "average_rating"]]
              .sort_values(key))

def same_description_dup(df: pd.DataFrame) -> pd.DataFrame:
    if "description" not in df.columns:
        return pd.DataFrame(columns=list(df.columns))
    g = (df.assign(_desc=df["description"].fillna(""))
            .groupby("_desc", dropna=False, sort=False))
    groups = []
    for desc, sub in g:
        if len(sub) <= 1:
            continue
        groups.append({
            "n_rows": len(sub),
            "sample_desc_80": (desc[:80] + "…") if isinstance(desc, str) and len(desc) > 80 else desc,
            "distinct_titles": sorted(map(str, set(sub["title"]))),
            "distinct_authors": sorted(map(str, set(sub["authors"]))),
            "years": sorted({int(y) for y in pd.to_numeric(sub["original_publication_year"], errors="coerce").dropna()}),
            "book_ids": sub["book_id"].tolist(),
        })
    return pd.DataFrame(groups).sort_values("n_rows", ascending=False, ignore_index=True)


def same_description_groups(df: pd.DataFrame, min_group: int = 2) -> pd.DataFrame:
    if "desc_hash" not in df.columns:
        raise ValueError("desc_hash missing. Run cleaning first.")
    g = df.groupby("desc_hash", dropna=False)
    big = g.filter(lambda x: len(x) >= min_group).copy()
    if big.empty:
        return pd.DataFrame(columns=[
            "desc_hash","n_rows","sample_desc_80","distinct_titles","distinct_authors","years","book_ids"
        ])
    out = (
        big.groupby("desc_hash", as_index=False)
          .apply(lambda x: pd.Series({
              "n_rows": len(x),
              "sample_desc_80": str(x["description"].iloc[0])[:80],
              "distinct_titles": sorted(x["title"].dropna().unique().tolist()),
              "distinct_authors": sorted(x["authors"].dropna().unique().tolist()),
              "years": sorted(pd.to_numeric(x["original_publication_year"], errors="coerce")
                              .dropna().astype(int).unique().tolist()),
              "book_ids": sorted(x["book_id"].astype(int).tolist()),
          }))
          .reset_index(drop=True)
          .sort_values("n_rows", ascending=False)
    )
    return out


def title_norm_author_overlap_dups(df: pd.DataFrame) -> pd.DataFrame:
    def _norm_title(s: str) -> str:
        import re
        s = (str(s) or "").lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    tmp = df.copy()
    tmp["title_norm"] = tmp["title"].apply(_norm_title)
    tmp["author_list"] = tmp["authors"].fillna("").astype(str).apply(lambda s: [a.strip().lower() for a in s.split(",") if a.strip()])
    out = []

    for tnorm, sub in tmp.groupby("title_norm"):
        if len(sub) <= 1:
            continue
        idxs = list(sub.index)
        auth_sets = {i: set(sub.loc[i, "author_list"]) for i in idxs}
        for i_pos in range(len(idxs)):
            for j_pos in range(i_pos + 1, len(idxs)):
                i, j = idxs[i_pos], idxs[j_pos]
                if auth_sets[i] & auth_sets[j]:
                    out.append({
                        "title_norm": tnorm,
                        "book_id_i": int(sub.at[i, "book_id"]),
                        "book_id_j": int(sub.at[j, "book_id"]),
                        "title_i": sub.loc[i, "title"],
                        "title_j": sub.loc[j, "title"],
                        "authors_i": sub.loc[i, "authors"],
                        "authors_j": sub.loc[j, "authors"],
                        "year_i": sub.loc[i, "original_publication_year"],
                        "year_j": sub.loc[j, "original_publication_year"],
                    })
    return pd.DataFrame(out)
