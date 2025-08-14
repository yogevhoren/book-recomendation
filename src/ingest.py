from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from src.config import init_logging
init_logging()
import logging
log = logging.getLogger(__name__)


REQUIRED_COLS = {
    "book_id", "title", "authors", "original_publication_year",
    "language_code", "average_rating", "image_url", "description",
}

DTYPES = {
    "book_id": "int64",
    "title": "string",
    "authors": "string",
    "original_publication_year": "float64", 
    "language_code": "string",
    "average_rating": "float64",
    "image_url": "string",
    "description": "string",
}

def load_books_csv(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at {p}")

    df = pd.read_csv(p, dtype=DTYPES, nrows=nrows, keep_default_na=True)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    log.info(f"Loaded {p.name} | rows={len(df):,} cols={len(df.columns)}")
    return df
