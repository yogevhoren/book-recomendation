from __future__ import annotations
import os, time, json
from typing import Dict, List, Optional
from pathlib import Path
import requests
import logging

log = logging.getLogger(__name__)

def _q(title: str, authors: str) -> str:
    t = str(title or "").strip().replace('"', '')
    a = str(authors or "").strip().replace('"', '')
    parts = []
    if t:
        parts.append(f'intitle:"{t}"')
    if a:
        parts.append(f'inauthor:"{a}"')
    return "+".join(parts) if parts else ""

def fetch_google_books_genres(title: str, authors: str, api_key: Optional[str] = None, timeout: float = 8.0) -> List[str]:
    query = _q(title, authors)
    if not query:
        log.warning("fetch_google_books_genres: empty query for title=%r authors=%r", title, authors)
        return []
    params = {"q": query, "maxResults": 3, "printType": "books", "projection": "lite"}
    if api_key:
        params["key"] = api_key
    r = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=timeout)
    if r.status_code != 200:
        log.warning("Google Books API error status=%d for query=%s", r.status_code, query)
        return []
    js = r.json() or {}
    items = js.get("items", []) or []
    for it in items:
        info = (it or {}).get("volumeInfo", {}) or {}
        log.info("Google Books API returned %d items for query=%s", len(items), query)
        log.info("Google Books API items columns=%s", list(info.keys()))
        if not info:
            log.info("Google Books API returned no volumeInfo for query=%s", query)
            return []
        cats = info.get("categories", []) or []
        if cats:
            log.info("Google Books API returned no categories for query=%s", query)
            return [str(c) for c in cats if c]
    return []

def hydrate_cache_google(df, cache_fp: str | Path, max_updates: int = 100, sleep: float = 0.2, api_key: Optional[str] = None) -> Dict[str, List[str]]:
    p = Path(cache_fp)
    cache: Dict[str, List[str]] = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            cache = json.load(f)
    updates = 0
    for i in range(len(df)):
        bid = str(int(df.loc[i, "book_id"]))
        if bid in cache:
            continue
        title = str(df.loc[i, "title"])
        authors = str(df.loc[i, "authors"])
        cats = fetch_google_books_genres(title, authors, api_key=api_key)
        if cats:
            log.info("Caching genres for book_id=%s title=%s authors=%s cats=%s", bid, title, authors, cats)
            cache[bid] = cats
            updates += 1
        else:
            log.info("No genres found for %s; not caching", bid)
        if updates % 50 == 0:
            log.info("hydrate_cache_google progress updates=%d", updates)
        if updates >= int(max_updates):
            break
        if sleep > 0:
            time.sleep(float(sleep))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    log.info("hydrate_cache_google done total_updates=%d size=%d path=%s", updates, len(cache), p)
    return cache
