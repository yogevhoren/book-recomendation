import numpy as np
import pandas as pd
import pytest

from src.recsys.genres import coarse_map, build_igf_weights
from src.recsys.eval import genre_at_k


# def test_coarse_map_basic_buckets():
#     assert coarse_map(["Urban Fantasy"]) == {"fantasy"}
#     assert coarse_map(["Science Fiction"]) == {"scifi"}
#     assert coarse_map(["Historical Romance"]) == {"romance"}
#     assert coarse_map(["Biography"]) == {"nonfiction"}
#     assert coarse_map(["Detective & Crime"]) == {"mystery_thriller"}
#     assert coarse_map(["Young Adult"]) == {"ya"}
#     assert coarse_map(["Middle Grade"]) == {"children"}
#     assert coarse_map(["Classics"]) == {"classics"}

def test_coarse_map_other_bucket_and_empty():
    assert coarse_map(["Weird Unmapped Label"]) == {"other"}
    assert coarse_map([]) == {"other"}
    assert coarse_map([""]) == {"other"}



def test_genre_at_k_overlap_and_no_overlap_unweighted():
    df = pd.DataFrame({
        "book_id": [1, 2, 3, 4],
        "title":   ["Seed", "B", "C", "D"],
    })


    cache = {
        "1": ["Epic Fantasy"],
        "2": ["Historical Romance"],
        "3": ["Urban Fantasy"],
        "4": ["Science Fiction"],
    }


    topk_ids = [1, 2, 3]  # NOTE: df index 1->book_id=2, 2->3; index 0 (seed) not included in real flow, but test is illustrative

    score = genre_at_k(seed_id=0, topk_ids=topk_ids, df=df, genre_cache=cache, igf=None, mode="combo")
    assert 0.0 <= score <= 1.0
    assert pytest.approx(score, rel=0, abs=1e-6) == (1.0 / 3.0)

def test_genre_at_k_all_non_overlap():
    df = pd.DataFrame({
        "book_id": [10, 11, 12],
        "title":   ["Seed", "B", "C"],
    })
    cache = {
        "10": ["Memoir"],          # -> nonfiction
        "11": ["Urban Fantasy"],   # -> fantasy
        "12": ["Space Opera"],     # -> scifi
    }

    score = genre_at_k(seed_id=0, topk_ids=[1, 2], df=df, genre_cache=cache, igf=None, mode="combo")
    assert score == 0.0

def test_genre_at_k_missing_cache_entries_count_as_zero():
    df = pd.DataFrame({
        "book_id": [100, 101, 102],
        "title":   ["Seed", "B", "C"],
    })
    cache = {
        "100": ["Urban Fantasy"],  # seed has fantasy
        "101": [],                 # missing categories -> treated as no overlap
        # "102" missing entirely
    }
    score = genre_at_k(seed_id=0, topk_ids=[1, 2], df=df, genre_cache=cache, igf=None, mode="combo")
    assert score == 0.0



def test_build_igf_rare_combo_gets_higher_weight_than_common_combo():
   
    cache = {
        "1": ["Urban Fantasy"],          # fantasy
        "2": ["Epic Fantasy"],           # fantasy
        "3": ["Fantasy", "Romance"],     # fantasy+romance
        "4": ["Fantasy", "Romance"],     # fantasy+romance (common)
        "5": ["Science Fiction"],        # scifi
        "6": ["Historical Romance"],     # romance
        "7": ["Historical Romance"],     # romance (common)
        "8": ["Historical Romance"],     # romance (common)
        "9": ["Mystery"],                # mystery_thriller
        "10": ["Fantasy", "Romance"],    # fantasy+romance (common)
    }
    igf = build_igf_weights(cache, mode="combo")
    w_common = igf.get(tuple(sorted({"fantasy", "romance"})), 0.0)
    w_rare = igf.get(tuple(sorted({"mystery_thriller"})), 0.0)  # singleton combo "mystery_thriller"
    assert 0.0 <= w_common <= 1.0
    assert 0.0 <= w_rare <= 1.0
    assert w_rare > w_common, "Rare combos should have higher IAF than common combos"

def test_genre_at_k_with_igf_downweights_common_and_upweights_rare():
    df = pd.DataFrame({
        "book_id": [1, 2, 3, 4],
        "title":   ["Seed", "B", "C", "D"],
    })
    # Seed: fantasy
    cache = {
        "1": ["Urban Fantasy"],      # fantasy
        "2": ["Fantasy", "Romance"], # fantasy+romance (assume common)
        "3": ["Mystery"],            # mystery_thriller (no overlap with fantasy)
        "4": ["Epic Fantasy"],       # fantasy (assume rarer than combo)
    }
    igf = build_igf_weights(cache, mode="combo")
    # topk contains one common overlap (fantasy+romance), one rare overlap (fantasy), one non-overlap
    topk_ids = [1, 2, 3]  # df indices 1,2,3 -> book_id 2,3,4
    s_unweighted = genre_at_k(seed_id=0, topk_ids=topk_ids, df=df, genre_cache=cache, igf=None, mode="combo")
    s_weighted = genre_at_k(seed_id=0, topk_ids=topk_ids, df=df, genre_cache=cache, igf=igf, mode="combo")
    assert 0.0 <= s_unweighted <= 1.0
    assert 0.0 <= s_weighted <= 1.0
    # Weighted score should differ from plain mean when combos have different frequencies
    assert not np.isclose(s_unweighted, s_weighted)



# def test_genre_at_k_not_always_one_regression_guard():
#     df = pd.DataFrame({
#         "book_id": [101, 102, 103, 104, 105],
#         "title":   ["Seed", "B", "C", "D", "E"],
#     })
#     cache = {
#         "101": ["Fantasy"],
#         "102": ["Romance"],
#         "103": ["Science Fiction"],
#         "104": ["Fantasy"],
#         "105": ["Nonfiction"],
#     }
#    
#     topk_ids = [1, 2, 3, 4]
#     s = genre_at_k(seed_id=0, topk_ids=topk_ids, df=df, genre_cache=cache, igf=None, mode="combo")
#     assert pytest.approx(s, rel=0, abs=1e-6) == 0.5
