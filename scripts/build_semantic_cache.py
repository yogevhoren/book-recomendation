import os, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
from src.clean.pipeline import clean_books_dataset

def series_hash(s):
    h = hashlib.sha1()
    for x in s.astype(str):
        h.update(x.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def main(run_tag="eda2", model_name="BAAI/bge-small-en-v1.5", batch_size=64):
    raw = pd.read_csv("data/raw/book.csv")
    df = clean_books_dataset(raw, drop_language_col=True)
    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
    payload_hash = series_hash(texts)
    root = Path("artifacts/semantic/bge-small-en-v1.5"); root.mkdir(parents=True, exist_ok=True)
    z_path = root / f"Z_{run_tag}.npy"
    m_path = root / f"manifest_{run_tag}.json"
    if z_path.exists() and m_path.exists():
        man = json.loads(m_path.read_text(encoding="utf-8"))
        if man.get("payload_hash") == payload_hash and man.get("model_name") == model_name and man.get("rows") == len(texts):
            print("semantic cache already up to date")
            return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    Z = model.encode(texts.tolist(), batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True, device=device)
    np.save(z_path, Z)
    m_path.write_text(json.dumps({"payload_hash": payload_hash, "model_name": model_name, "rows": int(len(texts))}, ensure_ascii=False), encoding="utf-8")
    print(f"semantic cache written: {z_path}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
