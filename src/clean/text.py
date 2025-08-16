from __future__ import annotations
import logging, re
from .patterns import DISCLAIMER_TRIGGER, DISCLAIMER_ENDMARK
from .constants import ENGLISH_STOPWORDS_STRICT
import hashlib
import unicodedata

log = logging.getLogger(__name__)

def strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def standardize_typography(text: str) -> str:
    if not isinstance(text, str):
        return ""
    table = str.maketrans({
        "’": "'", "‘": "'", "“": '"', "”": '"',
        "–": "-", "—": "-",
        "\u00A0": " ",
    })
    t = text.translate(table)
    t = re.sub(r"[\u0000-\u001F\u007F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def trim_disclaimer_prefix_if_present(desc: str) -> str:
    if not isinstance(desc, str) or not desc:
        return desc
    if not DISCLAIMER_TRIGGER.search(desc):
        return desc
    m_end = DISCLAIMER_ENDMARK.search(desc)
    if not m_end:
        log.info("Found disclaimer trigger without end marker; left description unchanged.")
        return desc
    return desc[m_end.end():].lstrip()

def _english_stopword_share(tokens, stopword_set):
    if not tokens:
        return 0.0
    sw = sum(1 for t in tokens if t in stopword_set)
    return sw / max(1, len(tokens))


def flag_suspected_non_english(
    text: str,
    *,
    min_tokens: int = 30,
    stopword_floor: float = 0.02,
    non_ascii_ratio_threshold: float = 0.40,
    w_stopword: float = 0.7,
    w_non_ascii: float = 0.3,
    combined_threshold: float = 0.7
) -> tuple[bool, dict]:
    tokens = [t.lower() for t in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)]

    stopword_share_strict = _english_stopword_share(tokens, ENGLISH_STOPWORDS_STRICT)
    non_ascii_ratio = 0.0
    letters = [c for c in text if c.isalpha()]
    if letters:
        non_ascii_ratio = sum(1 for c in letters if ord(c) > 127) / len(letters)

    if len(tokens) >= min_tokens:
        stopword_score = (1 - stopword_share_strict) / (1 - stopword_floor)
    else:
        stopword_score =  combined_threshold * 0.9
    ascii_score = (non_ascii_ratio / non_ascii_ratio_threshold)
    score = w_stopword * stopword_score + w_non_ascii * ascii_score
    flag = score >= combined_threshold
    return flag, {
        "stopword_share_strict": stopword_share_strict,
        "non_ascii_ratio": non_ascii_ratio,
        "score": score,
        "tokens": len(tokens),
    }

def _normalize_for_hash(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def stable_text_hash(text: str) -> str:
    norm = _normalize_for_hash(text)
    h = hashlib.blake2b(norm.encode("utf-8"), digest_size=6)
    return h.hexdigest()
