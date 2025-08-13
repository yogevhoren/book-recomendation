from __future__ import annotations
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

_word_re = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

def description_length(df: pd.DataFrame, col: str = "description") -> pd.Series:
    return df[col].fillna("").astype(str).apply(lambda s: len(_word_re.findall(s)))

def unusual_char_report(df: pd.DataFrame, col: str = "title", top_n: int = 30) -> pd.DataFrame:
    allowed = r"a-zA-Z0-9\s\.,:;!?'\-\(\)\[\]/&"
    pat = re.compile(rf"[^{allowed}]")
    chars = []
    for v in df[col].fillna("").astype(str):
        chars.extend(pat.findall(v))
    cnt = Counter(chars)
    items = cnt.most_common(top_n)
    return pd.DataFrame(items, columns=["char", "count"])

def vocab_size(df: pd.DataFrame, col: str = "description") -> int:
    vec = CountVectorizer(stop_words="english")
    vec.fit(df[col].fillna("").astype(str))
    return len(vec.vocabulary_)

def author_flip_candidates(df: pd.DataFrame, sample_n: int = 30) -> pd.DataFrame:
    if "authors" not in df.columns:
        return pd.DataFrame(columns=["book_id","authors"])
    has_comma = df["authors"].astype(str).str.contains(",", regex=False, na=False)
    sample = df.loc[has_comma, ["book_id", "title", "authors"]].head(sample_n).copy()
    return sample
