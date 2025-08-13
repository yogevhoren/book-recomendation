from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import save_and_show

def hist_numeric(df: pd.DataFrame, col: str, bins: int = 30, save_path: str | None = None, title: str | None = None):
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=bins, kde=True, ax=ax)
    ax.set_title(title or f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    save_and_show(fig, save_path)

def bar_topk(df: pd.DataFrame, col: str, k: int = 10, save_path: str | None = None, title: str | None = None):
    vc = df[col].value_counts(dropna=False).head(k)
    fig, ax = plt.subplots()
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title or f"Top {k}: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    save_and_show(fig, save_path)

def bar_all(df: pd.DataFrame, col: str, sort_desc: bool = True, save_path: str | None = None, title: str | None = None):
    vc = df[col].value_counts(dropna=False)
    if sort_desc:
        vc = vc.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4 + len(vc) * 0.12))
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title or f"Frequency: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    save_and_show(fig, save_path)
