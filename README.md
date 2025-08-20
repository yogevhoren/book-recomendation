# Book Recommendation System

Hybrid, unsupervised book recommender: TF‑IDF (bigrams) + semantic embeddings (BGE‑small) + authors (IAF) + engineered numerics + images (DINOv2). Built for interpretability, reproducibility, and laptop‑friendly runtime.

---

## Table of Contents
* [Overview](#Overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Requirements](#requirements)
* [Quick Start](#quick-start)
* [Configuration](#configuration)
* [Data](#data)
* \[Artifacts & Caching ("Artifactory")] (#artifacts--caching-artifactory)
* [How to Use (Code)](#how-to-use-code)
* [Evaluation](#evaluation)
* [Key APIs & Models](#key-apis--models)
* [Best Practices](#best-practices)
* [Troubleshooting](#troubleshooting)

---
# Overview
I built an unsupervised book recommender. Given a single seed book, it returns a short ranked list of similar recommended books. The focus is on clean data handling, interpretable features, a transparent hybrid ranking method, and a clear evaluation plan that does not require labels.

Problem Definition 
Input: One seed book from a corpus. Each book includes title, description, authors, publication year, average rating, and a cover image URL.
Output: A top 5 recommendation list that excludes the seed book and avoids near duplicates.
Representations (feature tracks):
1.	TF IDF text vectors built from title + description (bigrams).
2.	Semantic sentence embeddings using BGE small (compact and laptop friendly).
3.	Authors as a multi hot vector with Inverse Author Frequency (IAF) weights.
4.	Engineered numeric features (length of description, author counts, normalized rating and year, flags for modern and pre 1900).
5.	Image embeddings from DINOv2 (only when the book has a real cover).
Similarity per track: Cosine similarity between the seed’s vector and each candidate’s vector (all vectors are L2 normalized). This gives one similarity score per track.
Rank normalization and fusion: Within each track, convert similarities into percentile scores (0 to 1). Then compute a weighted average across the available tracks. If a track is missing for a book (for example, no image), weights are renormalized across the tracks that do exist.


---

## Features

* **Modular tracks:** lexical (TF‑IDF), semantic (BGE‑small), authors (IAF), numerics, image (DINOv2).
* **Rank‑percentile fusion:** per‑track cosine → percentiles → weighted blend.
* **Diversity controls:** MMR re‑rank (light), caps by author and identical description.
* **Explainability:** per‑modality contributions, top TF‑IDF terms, author overlaps, deltas in year/rating, cover images.
* **Reproducibility:** hash‑based caching to `artifacts/` with JSON manifests.

---

## Project Structure

```
src/
  config.py                 # seeds, paths, logging
  ingest.py                 # CSV loading & schema checks
  features_representation/
    pipeline.py             # fit_or_load_all: orchestrates all tracks
    lexical.py              # TF‑IDF corpus + vectorizer
    semantic.py             # BGE‑small embeddings
    metadata.py             # authors + IAF; numeric scalers and transform
    image.py                # DINOv2 cover embeddings, covers cache
  recsys/
    utils.py                # similarity, nearest table, caps, diagnostics
    metrics.py              # Genre@K (IGF), agreement, diversity
notebooks/
  10_eda1.ipynb            # exploration & cleaning
  20_eda2.ipynb            # engineered features & correlations
  30_fusion_tuning.ipynb   # weight search, MMR, diagnostics
  40_results_and_recs.ipynb# final results & explanations
artifacts/                  # cached outputs per track (see below)
```

> Folder names may vary; keep the relative roles consistent.

---

## Requirements

* Python **3.10–3.11** (recommended)
* OS: Linux/macOS/Windows; GPU optional (CUDA if available)

### Python packages

```
pandas
numpy
scikit-learn
scipy
transformers
sentence-transformers  # if used for BGE utility wrappers
torch
torchvision
timm
Pillow
umap-learn
requests
tqdm
```

> Create a `requirements.txt` with the above and install via `pip install -r requirements.txt`.

---

## Quick Start

```bash
# 1) Clone
git clone <YOUR_REPO_URL>.git book-recommender
cd book-recommender

# 2) Create & activate venv (example: Unix)
python -m venv .venv
source .venv/bin/activate

# 3) Install deps
pip install -U pip
pip install -r requirements.txt

# 4) Put the dataset
mkdir -p data/raw
cp path/to/book.csv data/raw/

# 5) (Optional) Set environment
# export PROJECT_ROOT=$(pwd)
# export BOOKS_CSV=$(pwd)/data/raw/book.csv

# 6) Run the notebooks in order
# notebooks/10_eda1.ipynb → 20_eda2.ipynb → 30_fusion_tuning.ipynb → 40_results_and_recs.ipynb
```

---

## Configuration

Edit `src/config.py`:

* **Paths**: `get_paths()` uses `PROJECT_ROOT` (optional) and `BOOKS_CSV` env var (optional). Defaults: `data/raw/book.csv`, `artifacts/` tree.
* **Logging**: `init_logging()` sets a sensible format.
* **Image params**: `IMG_H, IMG_W`, ImageNet mean/std.
* **Seed**: `SEED=42` (change if needed).

---

## Data

Expected CSV columns:

```
book_id, title, authors, original_publication_year,
language_code, average_rating, image_url, description
```

`src/ingest.py` enforces dtypes and required columns. Cleaning (whitespace, dedupe, description hashing, non‑English flagging) happens before feature extraction.

---

## Artifacts & Caching ("Artifactory")

Each stage writes compact artifacts to `artifacts/` with a manifest that includes input hash (`df_hash`), shapes, and key parameters.

Typical layout:

```
artifacts/
  tfidf/<run_tag>/
    vectorizer.pkl, tfidf_csr.npz, manifest.json
  semantic/<run_tag>/
    emb.npy, manifest.json
  metadata/
    authors_mlb.pkl, authors_iaf.npy, authors_manifest.json
    numeric_scalers.pkl, numeric_manifest.json
  image/<run_tag>/
    embeddings.npy, has_cover_mask.npy, manifest.json
    covers/  # cached downloaded covers (hashed filenames)
  manifest_all.json  # summary written by fit_or_load_all
```

**How it works:**

* Hashes of the input dataframe or key columns are stored in manifests.
* On rerun, if the hash and params match, the pipeline loads cached arrays instead of recomputing.
* Missing image covers are skipped (placeholders detected); weights are renormalized at fusion so items without images aren’t penalized.

---

## How to Use (Code)

Below are minimal examples. Adjust paths and params to your environment.

    df = pd.read_csv("data/raw/book.csv")
    df_clean = clean_books_dataset(df, drop_language_col=True)
    reps = fit_or_load_all(df_clean, artifacts_root=ARTIFACTS / "eda2", enable_image=True, run_tag_img="dinov2_vits14")
    genre_cache = ensure_genre_cache(df_clean, max_updates=100)
    best_w = eval_and_select_weights(df_clean, reps, genre_cache, use_mmr=True, lam=0.95, mmr_space=mmr_space)
    seed_idx = 0
    seed_book_id = int(df_clean.loc[seed_idx, "book_id"])
    post_year_alpha = 0.02
    mmr_on = False
    rec = serve_final(df_clean, reps, seed_book_id, weights=best_w, year_bias_weight=post_year_alpha, use_mmr=mmr_on, mmr_space=mmr_space)
```

> For production, add MMR re‑rank, author/desc caps, and an optional year bias (+0.05). Utilities for caps and tables exist in `src/recsys/utils.py`.

---

## Evaluation

* **Genre\@K (IGF‑weighted):** Rewards overlaps in rarer genres more than popular ones (fairness vs. best‑seller bias).
* **YearFlag\@K:** Same era bucket as the seed (modern vs. pre‑1900).
* **Agreement (Jaccard/Kendall):** Diagnostic overlap between tracks.
* **Diversity (HHI, year spread):** Avoid author monopolies; healthy temporal variety.
* **Stability:** Resampling‑based list stability.

Run the **Fusion** and **Final** notebooks to reproduce plots and tables.

---

## Key APIs & Models

* **BGE‑small embeddings**: `BAAI/bge-small-en-v1.5` (via `transformers` / `sentence-transformers`). Good cosine‑space retrieval quality with small memory/latency.
* **DINOv2 ViT‑S/14**: via **timm** + **torchvision** for cover embeddings (token/global pooling).
* **Open Library subjects** (optional): used to hydrate genre “pseudo‑labels” for Genre\@K.

---

## Best Practices

* Keep notebook runs deterministic: set `SEED` in `config.py` and any library RNGs.
* Use the **artifacts cache**; delete a specific subfolder to force recompute.
* Treat missing images as “no image track” — don’t zero‑fill; let fusion renormalize.
* Prefer **bigrams** in TF‑IDF; avoid trigrams unless you can afford the memory.
* Keep a small **year bias**; too large drifts relevance.
* Document fusion weights used in each experiment (commit `weights.json`).

---

## Troubleshooting

**timm ImportError** → `pip install timm` (or run CPU‑only with smaller batch).
**Pillow/IO errors** → skip broken images; the pipeline already handles placeholders.
**CUDA not found** → run CPU; reduce batch sizes for images and semantic embedding.
**Huge TF‑IDF memory** → reduce `max_features`, increase `min_df`, keep `ngram_range=(1,2)`.

---

**License:** add your preferred license.
**Maintainer:** add your name/email.
**Contributions:** PRs welcome (tests and docs required).
