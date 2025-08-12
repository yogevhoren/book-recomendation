# tests/test_ingest.py
from src.config import get_paths, ensure_dirs
from src.ingest import load_books_csv

def test_ingest_loads_and_has_required_cols(caplog):
    paths = get_paths()
    ensure_dirs(paths)

    assert paths["raw_csv"].exists(), (
        f"Expected CSV at {paths['raw_csv']}. "
        "Place file at data/raw/book.csv or set BOOKS_CSV."
    )

    with caplog.at_level("INFO"):
        df = load_books_csv(paths["raw_csv"])

    required = {
        "book_id","title","authors","original_publication_year",
        "language_code","average_rating","image_url","description"
    }
    missing = required - set(df.columns)
    assert not df.empty
    assert not missing, f"Missing cols: {missing}"
    assert any("Loaded" in r.message for r in caplog.records)