import logging, pytest, sys
from src.config import init_logging, set_seed
from pathlib import Path
import pandas as pd

@pytest.fixture(scope="session", autouse=True)
def _session_config():
    init_logging(logging.INFO)
    set_seed()

PROJ_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJ_ROOT / "src"
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

@pytest.fixture
def toy_df():
    data = dict(
        book_id=[1,2,3,4,5,6],
        title=[
            "Space Odyssey", "Deep Learning", "Ocean Tales",
            "Ancient Myths", "Modern Systems", "City Lights"
        ],
        authors=[
            "Clark, Arthur C.", "Ian Goodfellow, Yoshua Bengio",
            "Jane Roe", "John Doe", "Mary Major", "John Doe, Jane Roe"
        ],
        description=[
            "A journey through space and time; epic science fiction.",
            "Neural networks and optimization for modern AI.",
            "Stories by the sea, waves and wind.",
            "Legends and heroes from the ancient world.",
            "Distributed design for modern software systems.",
            "Urban stories about people and places."
        ],
        image_url=[
            "https://example.com/a.jpg",
            "https://example.com/b.jpg",
            "https://example.com/c.jpg",
            "https://example.com/d.jpg",
            "https://example.com/e.jpg",
            "https://example.com/f.jpg",
        ],
        average_rating=[4.2, 4.5, 3.9, 4.1, 4.0, 3.8],
        original_publication_year=[1968, 2016, 1999, 1875, 2008, 2010],
    )
    return pd.DataFrame(data)

@pytest.fixture
def artifacts_dir(tmp_path):
    d = tmp_path / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d

