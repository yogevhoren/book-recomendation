from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import logging

log = logging.getLogger(__name__)

def load_cache(fp: str | Path) -> Dict[str, List[str]]:
    p = Path(fp)
    if not p.exists():
        log.info("load_cache miss path=%s", p)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        js = json.load(f)
    log.info("load_cache ok n=%d path=%s", len(js), p)
    return js

def save_cache(fp: str | Path, cache: Dict[str, List[str]]) -> None:
    Path(fp).parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    log.info("save_cache ok n=%d path=%s", len(cache), fp)

_COARSE = {
    "fantasy": {"fantasy", "epic fantasy", "urban fantasy", "dragons", "magic", "mythical creatures", "fantasy fiction", "american fantasy fiction"},
    "romance": {"romance", "love stories", "historical romance", "contemporary romance", "fiction, romance, contemporary", "fiction, romance, western"},
    "mystery_thriller": {"mystery", "thriller", "crime", "detective", "suspense", "whodunit", "roman policier", "fiction, suspense", "fiction, thrillers", "fiction, mystery & detective, general"},
    "scifi": {"science fiction", "sci-fi", "space opera", "dystopia", "post-apocalyptic fiction", "fiction, science fiction, general", "fiction, science fiction, hard science fiction"},
    "horror": {"horror", "psychological horror", "supernatural horror", "ghost stories", "zombies", "fiction, horror", "horror tales", "horror stories"},
    "adventure": {"adventure", "action", "action-adventure", "quests", "exploration", "fiction, action & adventure"},
    "fiction": {"fiction", "literary fiction", "general fiction", "contemporary fiction", "domestic fiction", "fiction, family life", "fiction, historical", "fiction, psychological"},
    "historical": {"historical", "historical fiction", "historical romance", "period drama", "fiction, historical, general"},
    "biography": {"biography", "memoir", "autobiography", "personal memoirs", "biography & autobiography", "biography & autobiography / social scientists & psychologists"},
    "self_help": {"self-help", "personal development", "self-improvement", "motivational"},
    "poetry": {"poetry", "poems", "verse", "free verse poetry"},
    "graphic_novel": {"graphic novel", "comics", "manga", "illustrated books", "cartoons and comics", "comics & graphic novels"},
    "nonfiction": {"nonfiction", "history", "essays", "psychology / history", "comic books, strips, etc., history and criticism"},
    "ya": {"young adult", "ya", "teen fiction", "juvenile fiction", "fiction, coming of age"},
    "children": {"children", "kids", "middle grade", "picture book", "juvenile literature", "children's fiction"},
    "classics": {"classics", "classic", "literary classics", "classic literature"},
    "dystopian": {"dystopia", "post-apocalyptic", "future", "totalitarianism"},
    "social_issues": {"social issues", "race relations", "gender roles", "class differences", "interracial marriage", "social themes"},
    "war_military": {"war", "military", "world war", "historical war", "dwarfs", "award:newbery_award"},
    "health": {"nutrition", "birth control", "diet fads", "feeding behavior", "natural foods", "nutrition and dietetics"},
    "translations": {"tłumaczenia polskie", "translations into Spanish", "roman anglais", "powieść autobiograficzna amerykańska"},
    "other": set() 
}

def coarse_map(raw_genres: List[str]) -> Set[str]:
    res: Set[str] = set()
    for g in raw_genres or []:
        s = str(g).strip().lower()
        matched = False
        for k, vs in _COARSE.items():
            if k == "other":
                continue
            for v in vs:
                if v in s:
                    res.add(k)
                    matched = True
                    break
            if matched:
                break
        if not matched and s:
            res.add("other")
    if not res:
        res = {"other"}
    return res

def build_igf_weights(cache: Dict[str, List[str]], mode: str = "combo") -> Dict[str | Tuple[str, ...], float]:
    combos: Dict[str | Tuple[str, ...], int] = {}
    for _, val in cache.items():
        coarse = list(coarse_map(val))
        if mode == "combo":
            key = tuple(sorted(coarse))
            combos[key] = combos.get(key, 0) + 1
        else:
            for g in coarse:
                combos[g] = combos.get(g, 0) + 1
    N = sum(combos.values())
    out: Dict[str | Tuple[str, ...], float] = {}
    for k, df in combos.items():
        out[k] = float((N + 1.0) / (df + 1.0))
    m = max(out.values()) if out else 1.0
    out = {k: (v / m) for k, v in out.items()}
    log.info("build_igf_weights mode=%s unique=%d", mode, len(out))
    return out
