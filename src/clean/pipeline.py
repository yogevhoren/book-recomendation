from __future__ import annotations
import logging
import pandas as pd
from .text import strip_ws, standardize_typography, trim_disclaimer_prefix_if_present, flag_suspected_non_english
from .dedupe import author_list, consolidate_by_title_author_overlap
from .text import stable_text_hash

log = logging.getLogger(__name__)

def clean_books_dataset(
    df: pd.DataFrame,
    *,
    drop_language_col: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    start_rows = len(df)
    log.info(f"Cleaning started. Input rows={start_rows:,}.")

    for col in ("title", "authors", "description"):
        if col in df.columns:
            df[col] = df[col].map(strip_ws)
    log.info("Trimmed whitespace in title/authors/description.")

    df["author_list"] = df["authors"].map(author_list)
    log.info("Built author_list from authors.")

    before = len(df)
    df = consolidate_by_title_author_overlap(df)
    if len(df) == before:
        log.info("No rows removed by deduplication.")

    if "original_publication_year" in df.columns:
        df["original_publication_year"] = pd.to_numeric(df["original_publication_year"], errors="coerce")
        too_future = df["original_publication_year"] > 2026
        n_future = int(too_future.sum())
        if n_future:
            df.loc[too_future, "original_publication_year"] = pd.NA
            log.warning(f"Set NaN for {n_future} rows with year > 2026.")
    else:
        log.warning("Column 'original_publication_year' missing after dedup step.")

    if drop_language_col and "language_code" in df.columns:
        df = df.drop(columns=["language_code"])
        log.info("Dropped language_code column.")

    df["description"] = df["description"].map(standardize_typography)
    mask_trigger = df["description"].str.contains("not the actual book", case=False, na=False)
    mask_endmark = df["description"].str.contains("full copy of this great book\\.", case=False, na=False)
    mask_trim = mask_trigger & mask_endmark
    n_trim = int(mask_trim.sum())
    if n_trim:
        df.loc[mask_trim, "description"] = df.loc[mask_trim, "description"].map(trim_disclaimer_prefix_if_present)
        log.info(f"Trimmed disclaimer prefix through end marker for {n_trim} rows.")

    res = df["description"].apply(lambda s: flag_suspected_non_english(s))

    flags, diags = zip(*res) if len(res) else ([], [])
    df["desc_suspected_non_english"] = list(flags)

    n_flag = sum(flags)
    if n_flag:
        shares = [d.get("stopword_share_strict", None) for d in diags if isinstance(d, dict)]
        nonascii = [d.get("non_ascii_ratio", None) for d in diags if isinstance(d, dict)]
        log.info(
            f"Flagged suspected non-English: {n_flag} rows "
            f"(median strict-stopword={pd.Series(shares).median():.3f} | "
            f"median non-ascii={pd.Series(nonascii).median():.3f})"
        )
    else:
        log.info("No suspected non-English descriptions flagged.")

    df["desc_hash"] = df["description"].map(stable_text_hash)
    grp_sizes = df.groupby("desc_hash", dropna=False)["book_id"].transform("size")
    df["desc_group_size"] = grp_sizes.astype("int64")
    df["desc_is_shared"] = (df["desc_group_size"] > 1)
    log.info(
        "Description grouping added: shared groups=%d (of %d rows).",
        int((df["desc_is_shared"]).sum()),
        len(df)
    )

    log.info(f"Cleaning complete. Output rows={len(df):,}. Columns={list(df.columns)}.")
    return df.reset_index(drop=True)
