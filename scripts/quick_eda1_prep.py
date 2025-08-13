from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

from src.ingest import load_books_csv
from src.data.validation import basic_profile, null_report
from src.eda.text_stats import description_length, vocab_size, author_flip_candidates
from src.eda.dup_checks import exact_title_author_dups, same_description_groups, title_norm_author_overlap_dups

def main(csv_path: str, out_dir: str = "artifacts/eda1_prep"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_books_csv(csv_path)

    pq_path = out / "raw_books.parquet"
    df.to_parquet(pq_path, index=False)

    profile = basic_profile(df)
    (out / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    nulls = null_report(df)
    nulls.to_csv(out / "null_report.csv")
    desc_len = description_length(df)
    pd.DataFrame({"book_id": df["book_id"], "desc_len": desc_len}).to_csv(out / "desc_length.csv", index=False)

    try:
        vsize = vocab_size(df)
    except Exception:
        vsize = None
    (out / "vocab_size.txt").write_text(str(vsize), encoding="utf-8")

    exact_dups = exact_title_author_dups(df)
    exact_dups.to_csv(out / "dups_exact_title_authors.csv", index=False)

    same_desc = same_description_groups(df)
    same_desc.to_csv(out / "dups_same_description_groups.csv", index=False)

    near_dups = title_norm_author_overlap_dups(df)
    near_dups.to_csv(out / "dups_title_norm_author_overlap_pairs.csv", index=False)
    
    author_flips = author_flip_candidates(df, sample_n=30)
    author_flips.to_csv(out / "author_flip_candidates.csv", index=False)

    print(f"Wrote EDA1 prep artifacts to {out.resolve()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to book.csv")
    ap.add_argument("--out", default="artifacts/eda1_prep", help="Output dir for cached artifacts")
    args = ap.parse_args()
    main(args.csv, args.out)
