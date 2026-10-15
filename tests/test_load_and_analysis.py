"""Tests on small hand-made tables: no downloads, no network."""

import pandas as pd

from nfx import analysis as A
from nfx import load as L


def test_norm_and_split_title():
    assert L.norm("Amélie & Co.!") == "amelie and co"
    assert L.split_title("Bridgerton: Season 4 // Bridgerton: Temporada 4") == ("Bridgerton", "Bridgerton", "Season 4")
    assert L.split_title("His & Hers: Limited Series")[0] == "His & Hers"
    assert L.split_title("Squid Game: Season 3 // 오징어 게임: 시즌 3") == ("Squid Game", "오징어 게임", "Season 3")
    base, orig, _ = L.split_title("The Great Flood // 대홍수")
    assert base == "The Great Flood" and orig == "대홍수"  # kept: IMDb's originalTitle is often in Korean
    assert L.norm("대홍수") == "대홍수"
    assert L.norm("धुरंधर द रिवेंज") != ""  # Devanagari must not collapse to an empty key
    assert L.norm("$") == ""
    assert L.norm("모범택시3") == "모범택시3"  # not "3": digits alone must not decide the key
    assert L.split_title("Madagascar (2005)")[0] == "Madagascar" and L.year_hint("Madagascar (2005)") == 2005


def _imdb():
    return pd.DataFrame([
        {"tconst": "t1", "primaryTitle": "One Piece", "originalTitle": "One Piece", "startYear": 1999,
         "numVotes": 300000, "averageRating": 9.0, "genres": "Animation", "kind": "show"},
        {"tconst": "t2", "primaryTitle": "One Piece", "originalTitle": "One Piece", "startYear": 2023,
         "numVotes": 200000, "averageRating": 8.3, "genres": "Action", "kind": "show"},
        {"tconst": "t3", "primaryTitle": "Madagascar", "originalTitle": "Madagascar", "startYear": 2005,
         "numVotes": 500000, "averageRating": 6.9, "genres": "Animation,Family", "kind": "movie"},
        {"tconst": "t4", "primaryTitle": "Madagascar", "originalTitle": "Madagascar", "startYear": 1994,
         "numVotes": 900000, "averageRating": 5.0, "genres": "Drama", "kind": "movie"},
        {"tconst": "t6", "primaryTitle": "$", "originalTitle": "$", "startYear": 1971,
         "numVotes": 3000, "averageRating": 6.0, "genres": "Comedy", "kind": "movie"},
        {"tconst": "t5", "primaryTitle": "Stranger Things", "originalTitle": "Stranger Things", "startYear": 2016,
         "numVotes": 1400000, "averageRating": 8.6, "genres": "Drama", "kind": "show"},
    ])


def _engagement(rows):
    e = pd.DataFrame(rows)
    e["release_date"] = pd.to_datetime(e["release_date"])
    parts = e["title"].map(L.split_title)
    e["base_title"], e["original_title"] = parts.str[0], parts.str[1]
    e["year_hint"] = e["title"].map(L.year_hint)
    return e


def test_matching_rules():
    e = _engagement([
        {"title": "ONE PIECE: Season 2", "kind": "show", "release_date": "2026-03-10", "views": 1},
        {"title": "Madagascar (2005)", "kind": "movie", "release_date": None, "views": 1},
        {"title": "Stranger Things 5", "kind": "show", "release_date": "2025-11-26", "views": 1},
        {"title": "Not In IMDb", "kind": "movie", "release_date": None, "views": 1},
        {"title": "Dhurandhar The Revenge // धुरंधर द रिवेंज", "kind": "movie", "release_date": None, "views": 1},
    ])
    m = L.match_imdb(e, _imdb())
    assert m["tconst"].tolist()[:3] == ["t2", "t3", "t5"]  # live-action One Piece, 2005 film, bare season number
    assert pd.isna(m["tconst"].iloc[3])
    assert pd.isna(m["tconst"].iloc[4])  # regression: an empty normalised title used to match the film '$'



def test_concentration_and_genres():
    e = pd.DataFrame({"title": list("abcdefghij"), "views": [50, 20, 10, 5, 5, 4, 3, 1, 1, 1],
                      "kind": ["movie"] * 5 + ["show"] * 5, "netflix_release": [True] * 2 + [False] * 8,
                      "non_english": [False] * 9 + [True], "genres": ["Drama,Comedy"] * 5 + ["Family"] * 5,
                      "rating": [7.0] * 10})
    c = A.concentration(e)
    assert c["top_10pct_share"] == 0.5 and c["netflix_release_share"] == 0.7
    g = A.genres_long(e)
    assert g["views_share"].sum() == e["views"].sum()  # splitting across genres keeps the total
    assert round(A.kids_share(e)["kids_family_view_share"], 2) == 0.1


def test_taste_twins_jaccard():
    c = pd.DataFrame({"country_name": ["A", "A", "B", "B", "C"], "show_title": ["x", "y", "x", "y", "z"],
                      "week": pd.to_datetime(["2026-01-01"] * 5), "weekly_rank": [1, 2, 1, 2, 1]})
    t = A.taste_twins(c, since="2025-01-01")
    assert t.loc["A", "most_similar"] == "B" and t.loc["A", "jaccard"] == 1.0 and t.loc["C", "jaccard"] == 0.0
