from __future__ import annotations
import pandas as pd
from .schema import REQUIRED_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

def check_required_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def null_report(df: pd.DataFrame) -> pd.DataFrame:
    out = (df.isna().sum().rename("nulls").to_frame()
           .assign(total=len(df))
           .assign(pct=lambda t: (t["nulls"] / t["total"]).round(4))
           .sort_values("nulls", ascending=False))
    return out

def basic_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "numeric_summary": df[NUMERIC_COLUMNS].describe(include="all").to_dict() if set(NUMERIC_COLUMNS) <= set(df.columns) else {},
        "categorical_cardinality": {c: int(df[c].nunique(dropna=True)) for c in CATEGORICAL_COLUMNS if c in df.columns},
    }
