from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import timm

try:
    import timm 
except Exception as e:
    timm = None  

log = logging.getLogger(__name__)

try:
    from src.config import IMG_H, IMG_W, IMAGENET_MEAN, IMAGENET_STD  
    _TARGET_SIZE = (IMG_H, IMG_W)
    _MEAN = IMAGENET_MEAN
    _STD = IMAGENET_STD
except Exception:
    _TARGET_SIZE = (518, 518)  
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

def _safe_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _df_hash_for_images(df: pd.DataFrame) -> str:
    need = ["book_id", "image_url"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"_df_hash_for_images missing column: {c}")
    h = hashlib.sha256()
    for bid, url in zip(df["book_id"].astype(str), df["image_url"].astype(str)):
        h.update(bid.encode("utf-8")); h.update(b"\t"); h.update(url.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()

def is_placeholder_url(url: str) -> bool:
    if not isinstance(url, str):
        return True
    if url == "":
        return True
    u = url.strip().lower()
    return ("nophoto" in u)

def _letterbox_square(im: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    side = max(w, h)
    pad_w = (side - w) // 2
    pad_h = (side - h) // 2
    im_sq = ImageOps.expand(im, border=(pad_w, pad_h, side - w - pad_w, side - h - pad_h), fill=0)
    img_tf = transforms.Compose([
        transforms.Resize(target_size),   # Resize to (518, 518)
        transforms.ToTensor(),           # Convert to float tensor in [0,1]
        transforms.Normalize(mean=_MEAN, std=_STD),  # Normalize
    ])
    return img_tf(im_sq)

# def _to_tensor_normalized(im: Image.Image) -> torch.Tensor:
#     arr = np.asarray(im).astype(np.float32) / 255.0  
#     t = torch.from_numpy(arr).permute(2, 0, 1)       
#     mean = torch.tensor(_MEAN, dtype=torch.float32).view(3, 1, 1)
#     std = torch.tensor(_STD, dtype=torch.float32).view(3, 1, 1)
#     return (t - mean) / std

def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32)

@dataclass
class ImageArtifacts:
    out_dir: Path
    covers_dir: Path
    emb_path: Path
    mask_path: Path
    manifest_path: Path

    @staticmethod
    def in_dir(root: Path, run_tag: str) -> "ImageArtifacts":
        out = root / run_tag
        out.mkdir(parents=True, exist_ok=True)
        return ImageArtifacts(
            out_dir=out,
            covers_dir=Path("artifacts/image/covers"),
            emb_path=out / "embeddings.npy",
            mask_path=out / "has_cover_mask.npy",
            manifest_path=out / "manifest.json",
        )

    def save_manifest(
        self,
        *,
        book_ids: Sequence[int],
        df_hash: str,
        model_name: str,
        device: str,
        batch_size: int,
        shape: Tuple[int, int],
        pct_missing: float,
        skipped_placeholders: int,
        target_size: Tuple[int, int],
    ) -> None:
        man = {
            "book_ids": list(map(int, book_ids)),
            "df_hash": df_hash,
            "model_name": model_name,
            "device": device,
            "batch_size": int(batch_size),
            "shape": [int(shape[0]), int(shape[1])],
            "pct_missing": float(pct_missing),
            "skipped_placeholders": int(skipped_placeholders),
            "target_size": list(map(int, target_size)),
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=2)


def _get_dinov2(model_name: str = "vit_small_patch14_dinov2", device: Optional[str] = None):
    if timm is None:
        raise ImportError("timm is required for DINOv2 image embeddings (pip install timm).")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="token")
    model = model.eval().to(dev)

    @torch.no_grad()
    def encode_batch(batch: torch.Tensor) -> torch.Tensor:
        if batch.shape[1:] != (3, 518, 518):
            raise ValueError(f"Input dimensions are incorrect: {batch.shape[1:]}. Expected (3, 518, 518).")
        z = model(batch.to(dev))
        return F.normalize(z, dim=1)


    class _Encoder:
        def __init__(self, enc): self._enc = enc
        @torch.no_grad()
        def encode(self, batch: torch.Tensor) -> torch.Tensor:
            return self._enc(batch)

        @property
        def device(self) -> str:
            return str(dev)

    return _Encoder(encode_batch)

def download_covers(df: pd.DataFrame, out_dir: str | Path, timeout: int = 10) -> Tuple[List[str], np.ndarray, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    mask = np.zeros(len(df), dtype=bool)
    skipped_placeholders = 0

    for i, url in enumerate(tqdm(df["image_url"].astype(str).tolist(), desc="download covers", leave=False)):
        if not url or is_placeholder_url(url):
            paths.append("")
            if url and is_placeholder_url(url):
                skipped_placeholders += 1
            continue
        fn = _safe_md5(url) + ".jpg"
        fp = out_dir / fn
        if not fp.exists():
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                with open(fp, "wb") as f:
                    f.write(r.content)
            except Exception:
                paths.append("")
                continue
        paths.append(str(fp))
        mask[i] = True

    return paths, mask, skipped_placeholders

def _load_and_preprocess(path: str, target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
    if not path:
        return None
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        log.warning(f"Failed to load image {path}: {e}")
        return None
    im_tf = _letterbox_square(im, target_size)
    return im_tf

def fit_image_embeddings(
    df: pd.DataFrame,
    *,
    run_tag: str = "dinov2_vits14",
    artifacts_root: str | Path = "artifacts/image",
    covers_cache_dir: str | Path = "artifacts/image/covers",
    device: Optional[str] = None,
    batch_size: int = 64,
    force_recompute: bool = False,
    encoder_factory: Callable[[str, Optional[str]], object] = lambda name, dev: _get_dinov2(name, dev),
    model_name: str = "vit_small_patch14_dinov2",
) -> Tuple[np.ndarray, np.ndarray]:
    need = ["book_id", "image_url"]
    log.info("Fitting image embeddings with run_tag=%s", run_tag)
    for c in need:
        if c not in df.columns:
            raise ValueError(f"fit_image_embeddings missing column: {c}")
    arts = ImageArtifacts.in_dir(Path(artifacts_root), run_tag)
    log.info("Image artifacts directory: %s", arts.out_dir)
    df_hash = _df_hash_for_images(df)
    log.info("DataFrame hash for image features: %s", df_hash)

    if not force_recompute and arts.emb_path.exists() and arts.mask_path.exists() and arts.manifest_path.exists():
        try:
            log.debug("Checking cache for image embeddings in %s", arts.out_dir)
            with open(arts.manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            if man.get("df_hash") == df_hash and man.get("model_name") == model_name \
               and tuple(man.get("target_size", [])) == tuple(_TARGET_SIZE):
                emb = np.load(arts.emb_path)
                mask = np.load(arts.mask_path)
                log.info("Image embeddings [%s] cache hit: %s | shape=%s | missing=%.1f%%",
                         run_tag, arts.out_dir, emb.shape, 100.0 * (1.0 - mask.mean()))
                return emb.astype(np.float32), mask.astype(bool)
        except Exception as e:
            log.warning("Image manifest present but load failed (%s). Recomputing.", e)

    log.info("Downloading covers to %s", covers_cache_dir)
    # paths, has_mask, skipped_placeholders = download_covers(df, covers_cache_dir, timeout=10)
    # log.info(f"Mask after download_covers: {has_mask}")
    # valid_paths = [p for p, mask in zip(paths, has_mask) if mask]
    # valid_df = df[has_mask.astype(bool)]
    # n = len(paths)
    # log.info("Downloaded covers: %d | Skipped placeholders: %d", n, skipped_placeholders)
    # enc = encoder_factory(model_name, device)
    # dev = enc.device if hasattr(enc, "device") else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # log.info("Using device for image encoding: %s", dev)
    # feats: List[np.ndarray] = []
    # bs = int(batch_size)
    # valid = 0
    # for i in tqdm(range(0, len(valid_paths), bs), desc="encode covers", leave=False):
    #     batch_paths = valid_paths[i:i+bs]
    #     imgs = []
    #     keep_idx = [] 
    #     for j, p in enumerate(batch_paths):
    #         t = _load_and_preprocess(p, _TARGET_SIZE)
            
    #         if t is None:
    #             continue
    #         imgs.append(t)
    #         keep_idx.append(j)
    #     if len(imgs) == 0:
    #         feats.append(np.zeros((len(batch_paths), 1), dtype=np.float32))
    #         continue
    #     batch = torch.stack(imgs, dim=0)  
    #     with torch.no_grad():
    #         z = enc.encode(batch).detach().cpu().numpy().astype(np.float32)
    #         # if z is None:
    #         #     log.warning("Skipping batch due to unsupported feature shape.")
    #         #     feats.append(np.zeros((len(batch_paths), 1), dtype=np.float32))
    #         #     continue
    #     B = len(batch_paths)
    #     D = z.shape[1]
    #     chunk = np.zeros((B, D), dtype=np.float32)
    #     for k, jj in enumerate(keep_idx):
    #         chunk[jj] = z[k]
    #     feats.append(chunk)
    #     valid += len(keep_idx)
    # log.info("Encoded %d valid images out of %d total", valid, n)
    # emb = np.concatenate(feats, axis=0) if feats else np.zeros((n, 1), dtype=np.float32)
    # row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # nz = row_norms.squeeze(-1) > 0
    # emb[nz] = (emb[nz] / row_norms[nz])
    paths, has_mask, skipped_placeholders = download_covers(df, covers_cache_dir, timeout=10)
    log.info("Downloaded covers: %d | Skipped placeholders: %d", len(paths), skipped_placeholders)

    valid_idx = np.flatnonzero(has_mask.astype(bool))
    valid_paths = [paths[i] for i in valid_idx]
    n = len(paths)

    enc = encoder_factory(model_name, device)
    dev = enc.device if hasattr(enc, "device") else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device for image encoding: %s", dev)

    bs = int(batch_size)
    emb = None
    valid = 0

    for start in tqdm(range(0, len(valid_paths), bs), desc="encode covers", leave=False):
        batch_paths = valid_paths[start:start+bs]
        imgs = []
        keep_local = []
        for j, p in enumerate(batch_paths):
            t = _load_and_preprocess(p, _TARGET_SIZE)
            if t is None:
                continue
            imgs.append(t)
            keep_local.append(j)
        if not imgs:
            continue

        batch = torch.stack(imgs, dim=0)
        z = enc.encode(batch).detach().cpu().numpy().astype(np.float32)  # [B_keep, D]

        if emb is None:
            D = z.shape[1]
            emb = np.zeros((n, D), dtype=np.float32)

        global_batch_idx = valid_idx[start:start+bs]
        global_keep_idx = [int(global_batch_idx[j]) for j in keep_local]

        for k, g in enumerate(global_keep_idx):
            emb[g] = z[k]
        valid += len(global_keep_idx)

    log.info("Encoded %d valid images out of %d total", valid, n)

    if emb is None:
        emb = np.zeros((n, 1), dtype=np.float32)

    row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
    nz = (row_norms.squeeze(-1) > 0)
    emb[nz] = emb[nz] / row_norms[nz]

    log.info("Image embeddings shape: %s", emb.shape)
    log.info("Image embeddings dtype: %s", emb.dtype)
    np.save(arts.emb_path, emb.astype(np.float32))
    np.save(arts.mask_path, has_mask.astype(bool))
    log.info("Saved image embeddings to %s and mask to %s", arts.emb_path, arts.mask_path)
    pct_missing = float(1.0 - has_mask.mean())
    valid_idx = np.flatnonzero(has_mask.astype(bool))
    book_ids = df.iloc[valid_idx]["book_id"].tolist()
    arts.save_manifest(
        book_ids=book_ids,
        df_hash=df_hash,
        model_name=model_name,
        device=str(dev),
        batch_size=bs,
        shape=tuple(emb.shape),
        pct_missing=pct_missing * 100.0,
        skipped_placeholders=int(skipped_placeholders),
        target_size=_TARGET_SIZE,
    )
    log.info(
        "Image embeddings [%s] complete: shape=%s | missing=%.1f%% (placeholders skipped=%d) | saved to %s",
        run_tag, emb.shape, pct_missing * 100.0, skipped_placeholders, arts.out_dir
    )
    return emb, has_mask


def transform_image_embeddings(
    filepaths: Sequence[str],
    *,
    device: Optional[str] = None,
    batch_size: int = 64,
    encoder_factory: Callable[[str, Optional[str]], object] = lambda name, dev: _get_dinov2(name, dev),
    model_name: str = "vit_small_patch14_dinov2",
    target_size: Tuple[int, int] = _TARGET_SIZE,
) -> np.ndarray:
    enc = encoder_factory(model_name, device)
    feats: List[np.ndarray] = []
    bs = int(batch_size)
    n = len(filepaths)
    for i in range(0, n, bs):
        batch_paths = filepaths[i:i+bs]
        imgs = []
        keep = []
        for j, p in enumerate(batch_paths):
            t = _load_and_preprocess(p, target_size)
            if t is None:
                continue
            imgs.append(t); keep.append(j)
        if not imgs:
            feats.append(np.zeros((len(batch_paths), 1), dtype=np.float32))
            continue
        batch = torch.stack(imgs, dim=0)
        with torch.no_grad():
            z = enc.encode(batch).detach().cpu().numpy().astype(np.float32)
        B, D = len(batch_paths), z.shape[1]
        chunk = np.zeros((B, D), dtype=np.float32)
        for k, jj in enumerate(keep):
            chunk[jj] = z[k]
        feats.append(chunk)

    emb = np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    row_norms = np.linalg.norm(emb, axis=1, keepdims=True)
    nz = (row_norms.squeeze(-1) > 0)
    emb[nz] = (emb[nz] / row_norms[nz])
    return emb
