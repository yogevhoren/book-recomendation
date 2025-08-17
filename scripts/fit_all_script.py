import pandas as pd
import logging, sys, os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
ROOT = Path(os.getcwd()).parent
print(f"ROOT: {ROOT}")
sys.path.append(str(ROOT / "src"))
sys.path.append('..')
from src.clean.pipeline import clean_books_dataset
from src.features_representation.pipeline import fit_or_load_all

log = logging.getLogger(__name__)

ART = Path(ROOT / "artifacts/scripts") 

df = pd.read_csv("data/raw/book.csv")
df = clean_books_dataset(df, drop_language_col=True)
reps = fit_or_load_all(df, artifacts_root=ART)
log.info("----------")
log.info("Z_img: %s, mask share: %s", reps.Z_img.shape, float(reps.img_mask.mean()))
