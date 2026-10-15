import numpy as np
import pandas as pd

from nfx import recommend as R


def _meta():
    return pd.DataFrame({
        "base_title": ["Alpha", "Beta", "Gamma", "Delta"],
        "genres": ["Drama,Horror", "Drama,Horror", "Comedy", "Comedy"],
        "directors": [["Dir A"], ["Dir A"], ["Dir B"], []],
        "kind": ["movie", "movie", "movie", "show"], "imdb_year": [2020, 2021, 1999, 2024],
        "non_english": [False, False, False, True],
    }, index=["alpha", "beta", "gamma", "delta"])


def test_content_similarity_prefers_same_genre_and_director():
    x = R.content_matrix(_meta())
    sim = x @ x.T
    assert np.allclose(np.diag(sim), 1)
    assert sim[0, 1] > sim[0, 2] > sim[0, 3]


def test_because_you_watched_and_evaluate_run():
    weeks = pd.DataFrame([[3, 2, 0, 0], [2, 3, 0, 1], [0, 0, 4, 2]], index=["c1", "c2", "c3"],
                         columns=["alpha", "beta", "gamma", "delta"])
    recs = R.because_you_watched("Alpha", _meta(), weeks, n=2)
    assert recs["title"].iloc[0] == "Beta"
    # evaluate() needs 20+ titles per country; with tiny data it just reports no countries
    ev = R.evaluate(_meta(), weeks)
    assert set(ev.index) == {"popularity", "content", "collaborative", "hybrid"}
