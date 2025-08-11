from __future__ import annotations
from pathlib import Path
import logging, sys, os, random
import numpy as np
import torch 


# ---- Reproducibility ----
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)                
    if torch.cuda.is_available():           
        torch.cuda.manual_seed_all(seed)

# ---- Visual backbones ----
IMG_H, IMG_W = 518, 518
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---- Paths (dict-based) ----
def get_paths(root: Path | None = None, raw_csv: Path | None = None) -> dict[str, Path]:
    root = root or Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    paths = {
        "root": root,
        "data_raw": root / "data" / "raw",
        "data_interim": root / "data" / "interim",
        "data_processed": root / "data" / "processed",
        "artifacts": root / "artifacts",
        "tfidf": root / "artifacts" / "tfidf",
        "metadata": root / "artifacts" / "metadata",
        "semantic": root / "artifacts" / "semantic",
        "image": root / "artifacts" / "image",
        "raw_csv": raw_csv or Path(os.getenv("BOOKS_CSV", root / "data" / "raw" / "book.csv")),
    }
    return paths

def ensure_dirs(paths: dict[str, Path]) -> None:
    for key in ("data_raw", "data_interim", "data_processed", "tfidf", "metadata", "semantic", "image"):
        paths[key].mkdir(parents=True, exist_ok=True)


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(module)s:%(funcName)s | %(message)s"

def init_logging(level=logging.INFO):
    root = logging.getLogger()
    # prevent duplicate handlers if re-run in notebooks
    for h in list(root.handlers):
        root.removeHandler(h)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(h)
    root.setLevel(level)

