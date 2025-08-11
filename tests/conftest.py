# tests/conftest.py
import logging, pytest
from src.config import init_logging, set_seed

@pytest.fixture(scope="session", autouse=True)
def _session_config():
    # One-time, for all tests
    init_logging(logging.INFO)
    set_seed()
