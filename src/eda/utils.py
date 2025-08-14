from __future__ import annotations
import matplotlib.pyplot as plt

def set_eda_style():
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "axes.grid": True,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
    })

def save_and_show(fig, save_path: str | None = None, tight: bool = True):
    plt.show()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
