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

def plot_2d_embedding(Z: np.ndarray, labels: List[str], method: str = "umap", title: str = "Embedding (2D)"):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42)
            coords = reducer.fit_transform(Z)
        except Exception:
            method = "pca"
    if method == "pca":
        coords = PCA(n_components=2, random_state=42).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels))
    for g in unique:
        m = [i for i, lab in enumerate(labels) if lab == g]
        ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.6, label=g)
    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.legend(markerscale=2, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    return fig, ax
