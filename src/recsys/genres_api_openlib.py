from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import logging, time, json, requests

log = logging.getLogger(__name__)

BASE = "https://openlibrary.org"

def _search_query(title: str, authors: str) -> Dict[str, str]:
    q_parts = []
    t = (title or "").strip()
    a = (authors or "").strip()
    if t:
        q_parts.append(t)
    if a:
        q_parts.append(f'author:"{a}"')
    # fields keeps payload small and guarantees subjects back when present
    return {
        "q": " ".join(q_parts),
        "fields": "key,title,author_name,first_publish_year,subject,subject_facet,edition_key",
        "limit": "3",
    }

def fetch_openlib_subjects(title: str, authors: str, timeout: float = 8.0) -> List[str]:
    params = _search_query(title, authors)
    r = requests.get(f"{BASE}/search.json", params=params, timeout=timeout)
    if r.status_code != 200:
        log.debug("openlib search http=%s title=%s authors=%s", r.status_code, title, authors)
        return []
    js = r.json() or {}
    docs = js.get("docs", []) or []
    for d in docs:
        subs = d.get("subject_facet") or d.get("subject") or []
        if subs:
            return [s for s in subs if isinstance(s, str) and s.strip()]
    return []

def hydrate_cache_openlib(df, cache_fp: str | Path, max_updates: int = 100, sleep: float = 0.2) -> Dict[str, List[str]]:
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
        subs = fetch_openlib_subjects(title, authors)
        if subs:                      
            cache[bid] = subs
            updates += 1
            if updates % 50 == 0:
                log.info("hydrate_cache_openlib progress updates=%d", updates)
            if updates >= int(max_updates):
                break
            if sleep > 0:
                time.sleep(float(sleep))
        else:
            log.debug("No subjects for %s (%s)", bid, title)

    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    log.info("hydrate_cache_openlib done updates=%d total_cached=%d path=%s", updates, len(cache), p)
    return cache
