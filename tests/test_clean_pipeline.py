# tests/test_clean_pipeline.py
import pandas as pd
import numpy as np
import pytest

from src.clean.pipeline import clean_books_dataset


def test_clean_schema_and_invariants():
    df_in = pd.DataFrame({
        "book_id": [1, 2],
        "title": [" A Tale  of Two Cities ", "Les Misérables"],
        "authors": ["Charles Dickens", "Victor Hugo"],
        "original_publication_year": [1859, 1862],
        "language_code": ["en-GB", "eng"],
        "average_rating": [4.11, 4.32],
        "image_url": ["u1", "u2"],
        "description": ["Classic novel.", "Roman classique."],
    })
    out = clean_books_dataset(df_in, drop_language_col=True)

    must_have = {
        "book_id", "title", "authors", "author_list",
        "original_publication_year", "average_rating", "image_url",
        "description", "desc_suspected_non_english"
    }
    assert must_have.issubset(out.columns), f"Missing: {must_have - set(out.columns)}"

    assert "language_code" not in out.columns

    forbidden = {"authors_norm", "desc_len_words", "is_desc_empty", "year_norm"}
    assert not (forbidden & set(out.columns)), f"Unexpected feature cols in cleaning: {forbidden & set(out.columns)}"

    assert set(out["book_id"]) == set(df_in["book_id"])




def test_clean_no_mutation_and_idempotence():
    df_in = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["  Title  ", "Title"],
        "authors": ["A, B", "A, B"],
        "original_publication_year": [2010, 2008],
        "language_code": ["en", "en"],
        "average_rating": [4.0, 5.0],
        "image_url": ["u1", "u2"],
        "description": ["desc", "desc"],
    })
    df_snapshot = df_in.copy(deep=True)

    out1 = clean_books_dataset(df_in, drop_language_col=True)

    pd.testing.assert_frame_equal(df_in, df_snapshot, check_dtype=True, check_like=False)

    out2 = clean_books_dataset(out1, drop_language_col=True)
    pd.testing.assert_frame_equal(out1, out2, check_dtype=True, check_like=False)


def test_clean_dedupe_merge_and_no_merge_and_transitive():
    df_in = pd.DataFrame({
        "book_id": [10, 11,
                    20, 21,
                    30, 31, 32],
        "title": [
            "The Book — 1", "The Book 1",
            "Same Title", "Same Title",
            "Chain Book", "Chain Book", "Chain Book"
        ],
        "authors": [
            "Alice, Bob", "Bob, Carol",
            "X, Y", "Z",
            "Ann, Ben", "Ben, Cam", "Cam, Dan"
        ],
        "original_publication_year": [2010, 2008, 2005, 2007, 2003, 2002, 2004],
        "language_code": ["en"]*7,
        "average_rating": [4.0, 5.0, 3.0, 4.0, 4.2, 4.6, 4.8],
        "image_url": ["u"]*7,
        "description": ["d"]*7,
    })

    out = clean_books_dataset(df_in, drop_language_col=True)

    merged_group = out[out["title"].str.contains("The Book", na=False)]
    assert len(merged_group) == 1

    row = merged_group.iloc[0]
    pair = df_in[df_in["title"].str.contains("The Book", na=False)]
    expected_pair_id = pair.loc[pair["original_publication_year"].idxmin(), "book_id"]
    assert row["book_id"] == expected_pair_id
    authors_set = {a.strip().lower() for a in row["authors"].split(",")}
    assert authors_set == {"alice", "bob", "carol"}
    assert pytest.approx(row["average_rating"], rel=1e-6) == 4.5

    same_title = out[out["title"] == "Same Title"]
    assert len(same_title) == 2

    chain = out[out["title"] == "Chain Book"]
    assert len(chain) == 1

    triplet = df_in[df_in["title"] == "Chain Book"]
    expected_chain_id = triplet.loc[triplet["original_publication_year"].idxmin(), "book_id"]
    assert chain.iloc[0]["book_id"] == expected_chain_id


def test_clean_disclaimer_trim_rule():
    desc_trim = (
        "WARNING: This is NOT the actual book. Please read carefully. "
        "Also, not the actual book again for emphasis. "
        "Full copy of this great book. After this marker comes the real summary of the novel with punctuation—kept!"
    )
    desc_no_end = "Notice: not the actual book; study guide only. No end marker present here."

    df_in = pd.DataFrame({
        "book_id": [1, 2],
        "title": ["T1", "T2"],
        "authors": ["A", "B"],
        "original_publication_year": [2000, 2001],
        "language_code": ["en", "en"],
        "average_rating": [4.0, 4.0],
        "image_url": ["u1", "u2"],
        "description": [desc_trim, desc_no_end],
    })

    out = clean_books_dataset(df_in, drop_language_col=True)

    d1 = out.loc[out["book_id"] == 1, "description"].iloc[0]
    assert d1.lower().startswith("after this marker comes the real summary")
    assert "-" in d1 or "—" in d1

    d2 = out.loc[out["book_id"] == 2, "description"].iloc[0]
    assert "Notice:" in d2 or "notice:" in d2
    assert "No end marker present" in d2

def test_clean_non_english_flag():
    english_long = (
        "This is a long English paragraph that should not be flagged as non English. "
        "It contains plenty of common words and phrases to ensure that the stopword ratio is high enough "
        "to pass the conservative threshold for language detection without issues."
    )

    spanish_long = (
        "Este es un párrafo largo en español que no debería considerarse inglés. "
        "Incluye muchas palabras comunes y construcciones típicas del idioma para que la proporción "
        "de palabras funcionales en inglés sea muy baja y el texto sea marcado como no inglés."
    )

    cyrillic_long = (
        "Это длинный абзац на русском языке, написанный кириллицей, "
        "который должен быть помечен как не английский."
        " Он содержит много слов и фраз, которые не являются английскими, "
        "чтобы убедиться, что соотношение стоп-слов низкое и текст не будет ошибочно помечен как английский."
    )

    english_short = "Short English text."

    cyrillic_short = "Краткий текст на кириллице."

    df_in = pd.DataFrame({
        "book_id": [1, 2, 3, 4, 5],
        "title": ["E", "S", "R", "R2", "E2"],
        "authors": ["A", "B", "C", "D", "E"],
        "original_publication_year": [2000, 2001, 2002, 2003, 2004],
        "language_code": ["en", "es", "ru", "es", "en"],
        "average_rating": [4.0, 4.0, 4.0, 4.0, 4.0],
        "image_url": ["u1", "u2", "u3", "u4", "u5"],
        "description": [english_long, spanish_long, cyrillic_long, cyrillic_short, english_short],
    })

    out = clean_books_dataset(df_in, drop_language_col=True)

    flags = dict(zip(out["book_id"], out["desc_suspected_non_english"]))

    assert flags[1] is False, "English paragraph should NOT be flagged"
    assert flags[3] is True, "cyrillic long (non-Latin) should be flagged"
    assert flags[2] is True, "Spanish long (Latin but non-English) should be flagged via low EN stopwords"
    assert flags[4] is True, "Cyrillic short snippet should be flagged"
    assert flags[5] is False, "Short English text should NOT be flagged"
