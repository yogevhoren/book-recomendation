from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, List, Sequence, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  

log = logging.getLogger(__name__)

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Expected 2D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)

def _df_hash_for_semantic(df: pd.DataFrame) -> str:
    log.info("Computing DataFrame hash for semantic features")
    need = ["book_id", "title", "description", "desc_suspected_non_english"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"_df_hash_for_semantic missing column: {c}")
    joined = (
        df["book_id"].astype(str).fillna("")
        + "\u241F" + df["title"].astype(str).fillna("")
        + "\u241F" + df["description"].astype(str).fillna("")
        + "\u241F" + df["desc_suspected_non_english"].astype(str)
    ).tolist()
    h = sha256()
    for row in joined:
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

@dataclass
class SemanticArtifacts:
    out_dir: Path
    emb_path: Path
    manifest_path: Path

    @staticmethod
    def in_dir(base: Path) -> "SemanticArtifacts":
        base.mkdir(parents=True, exist_ok=True)
        return SemanticArtifacts(
            out_dir=base,
            emb_path=base / "embeddings.npy",      
            manifest_path=base / "manifest.json",
        )

    def save_manifest(
        self,
        *,
        book_ids: Sequence[int],
        df_hash: str,
        model_name: str,
        normalize: bool,
        batch_size: int,
        shape: Tuple[int, int],
        non_en_ratio: float,
    ) -> None:
        man = {
            "book_ids": list(map(int, book_ids)),
            "df_hash": df_hash,
            "model_name": model_name,
            "normalize": bool(normalize),
            "batch_size": int(batch_size),
            "shape": list(map(int, shape)),
            "non_english_ratio": float(non_en_ratio),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)

def build_semantic_corpus(df: pd.DataFrame) -> List[str]:
    log.info("Building semantic corpus from DataFrame with %d rows", len(df))
    need = ["title", "description"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"build_semantic_corpus missing column: {c}")
    titles = df["title"].astype(str).fillna("")
    descs = df["description"].astype(str).fillna("")
    texts = [f"{t}\n{d}".strip() for t, d in zip(titles, descs)]
    return texts


def _get_encoder(model_name: str, device: str | None):
    model = SentenceTransformer(model_name, device=device or "cpu")
    return model


def embed_bge_m3(
    texts: Sequence[str],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    device: str | None = None,
    normalize: bool = True,
    encoder_factory: Callable[[str, str | None], object] = _get_encoder,
) -> np.ndarray:
    log.info("Embedding %d texts with model %s", len(texts), model_name)
    if not isinstance(texts, (list, tuple)):
        texts = list(texts)
    if len(texts) == 0:
        return np.zeros((0, 1024), dtype=np.float32) 
    encoder = encoder_factory(model_name, device)
    emb = encoder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,  
        show_progress_bar=False,
    ).astype(np.float32)

    if normalize:
        emb = _l2_normalize_rows(emb)
    return emb


def fit_semantic(
    df: pd.DataFrame,
    *,
    run_tag: str = "BAAI/bge-small-en-v1.5",
    artifacts_root: str | Path = "artifacts/semantic",
    force_recompute: bool = False,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    device: str | None = None,
    normalize: bool = True,
    encoder_factory: Callable[[str, str | None], object] = _get_encoder,
) -> np.ndarray:
    log.info("Fitting semantic embeddings with run_tag=%s", run_tag)
    need = ["book_id", "title", "description", "desc_suspected_non_english"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"fit_semantic missing column: {c}")

    outdir = Path(artifacts_root) / run_tag
    arts = SemanticArtifacts.in_dir(outdir)
    df_hash = _df_hash_for_semantic(df)
    log.info("DataFrame hash for semantic features: %s", df_hash)
    if not force_recompute and arts.emb_path.exists() and arts.manifest_path.exists():
        log.info("Checking cache for semantic embeddings in %s", outdir)
        try:
            with open(arts.manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if (
                man.get("df_hash") == df_hash
                and man.get("model_name") == model_name
                and bool(man.get("normalize", True)) == bool(normalize)
            ):
                emb = np.load(arts.emb_path).astype(np.float32)
                log.info("Semantic [%s] cache hit → %s | shape=%s", run_tag, outdir, emb.shape)
                return emb
        except Exception as e:
            log.warning("Semantic [%s] manifest present but load failed (%s). Recomputing.", run_tag, e)

    texts = build_semantic_corpus(df)
    non_en_ratio = float(np.mean(df["desc_suspected_non_english"].astype(bool))) if len(df) else 0.0
    log.info("Non-English ratio in descriptions: %.1f%%", 100 * non_en_ratio)
    log.info("Total texts to embed: %d", len(texts))
    emb = embed_bge_m3(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        encoder_factory=encoder_factory,
    )
    log.info("Embedding complete: shape=%s", emb.shape)
    log.info("Saving semantic embeddings to %s", arts.emb_path)
    np.save(arts.emb_path, emb.astype(np.float32))
    arts.save_manifest(
        book_ids=df["book_id"].tolist(),
        df_hash=df_hash,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        shape=emb.shape,
        non_en_ratio=non_en_ratio,
    )
    log.info("Semantic [%s] fit complete: shape=%s | saved to %s", run_tag, emb.shape, outdir)
    return emb


def transform_semantic(
    texts: Sequence[str],
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    device: str | None = None,
    normalize: bool = True,
    encoder_factory: Callable[[str, str | None], object] = _get_encoder,
) -> np.ndarray:
    
    return embed_bge_m3(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        encoder_factory=encoder_factory,
    )
