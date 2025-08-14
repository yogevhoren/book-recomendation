from __future__ import annotations

REQUIRED_COLUMNS = [
    "book_id",
    "title",
    "authors",
    "original_publication_year",
    "language_code",
    "average_rating",
    "image_url",
    "description",
]

NUMERIC_COLUMNS = [
    "original_publication_year",
    "average_rating",
]

TEXT_COLUMNS = [
    "title",
    "authors",
    "description",
    "image_url",
]

CATEGORICAL_COLUMNS = [
    "language_code",
]
