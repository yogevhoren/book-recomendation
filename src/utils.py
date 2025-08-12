# src/utils.py
from contextlib import contextmanager
import time, logging

@contextmanager
def duration(msg: str):
    log = logging.getLogger(__name__)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t0) * 1000
        log.info(f"{msg} (took {ms:.1f} ms).")
