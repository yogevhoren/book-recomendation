from __future__ import annotations
import re


DISCLAIMER_TRIGGER = re.compile(r"not\s+the\s+actual\s+book", re.IGNORECASE)
DISCLAIMER_ENDMARK = re.compile(r"full copy of this great book\.", re.IGNORECASE)

__all__ = ["DISCLAIMER_TRIGGER", "DISCLAIMER_ENDMARK"]
