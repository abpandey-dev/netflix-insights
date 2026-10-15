"""Three small recommenders, built from public data, and a fair test of which works best.

  content        "similar to what you watched": cosine similarity of genre, director, decade, type, language
  collaborative  "watched by the same audiences": titles that chart in the same countries look alike. This is
                 the item-to-item idea behind "Because you watched X", with countries standing in for viewers
                 (Netflix's real system learns from members' individual viewing, which is private)
  hybrid         a weighted blend of the two
  popularity     baseline: recommend what charts in the most countries

Test: each country's set of Top 10 titles is a "viewing history". Hide 20% of it, recommend 20 titles from the
rest, and count how many hidden titles come back (recall@20). Collaborative scores for a country are always
computed from the *other* countries, so nothing leaks from the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfx.load import norm


def build_items(e: pd.DataFrame, crew: pd.DataFrame, c: pd.DataFrame, since: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Titles that charted in some country since `since` and matched to IMDb, plus a country x title matrix."""
    recent = c[c["week"] >= since].copy()
    recent["key"] = recent["show_title"].map(norm)
    meta = e.dropna(subset=["tconst"]).copy()
    meta["key"] = meta["base_title"].map(norm)
    meta = meta.sort_values("views", ascending=False).drop_duplicates("key")
    meta = meta[meta["key"].isin(set(recent["key"]))]
    dirs = crew.groupby("tconst")["director"].apply(list)
    meta["directors"] = meta["tconst"].map(dirs).map(lambda d: d if isinstance(d, list) and len(d) <= 3 else [])
    meta = meta.set_index("key")
    weeks = recent[recent["key"].isin(meta.index)].groupby(["country_name", "key"])["week"].nunique().unstack(
        fill_value=0)
    return meta, weeks


def content_matrix(meta: pd.DataFrame) -> np.ndarray:
    """Binary feature vectors: genres, directors (films), decade, movie/show, English/non-English; L2-normalised."""
    feats = []
    for _, r in meta.iterrows():
        f = {f"g:{g}" for g in str(r["genres"]).split(",") if g and g != "nan"}
        f |= {f"d:{d}" for d in (r["directors"] if r["kind"] == "movie" else [])}
        if pd.notna(r["imdb_year"]):
            f.add(f"y:{int(r['imdb_year']) // 10 * 10}")
        f.add(f"k:{r['kind']}")
        f.add("l:non_english" if r["non_english"] else "l:english")
        feats.append(f)
    vocab = {f: i for i, f in enumerate(sorted(set().union(*feats)))}
    x = np.zeros((len(feats), len(vocab)))
    for i, f in enumerate(feats):
        x[i, [vocab[k] for k in f]] = 1.0
        # genres and directors say more than "it's a movie": weight them up
        for k in f:
            if k[:2] in ("g:", "d:"):
                x[i, vocab[k]] = 2.0 if k.startswith("d:") else 1.5
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _normalise_cols(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=0, keepdims=True)
    n[n == 0] = 1
    return m / n


def scores(history: np.ndarray, sim: np.ndarray) -> np.ndarray:
    """Average similarity of every item to the items in the history (indices)."""
    return sim[:, history].mean(axis=1)


def evaluate(meta: pd.DataFrame, weeks: pd.DataFrame, k: int = 20, hide: float = 0.2, seed: int = 0,
             alpha: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    items = list(weeks.columns)
    meta = meta.loc[items]
    content_sim = content_matrix(meta) @ content_matrix(meta).T
    presence = (weeks.values > 0).astype(float)  # countries x items
    results = {"popularity": [], "content": [], "collaborative": [], "hybrid": []}
    for ci, country in enumerate(weeks.index):
        have = np.flatnonzero(presence[ci])
        if len(have) < 20:
            continue
        hidden = rng.choice(have, size=max(1, int(len(have) * hide)), replace=False)
        known = np.setdiff1d(have, hidden)
        others = np.delete(presence, ci, axis=0)  # leave this country out of the collaborative signal
        cf = _normalise_cols(others)
        cf_sim = cf.T @ cf
        pop = others.sum(axis=0)
        method_scores = {
            "popularity": pop.astype(float),
            "content": scores(known, content_sim),
            "collaborative": scores(known, cf_sim),
        }
        z = lambda s: (s - s.mean()) / (s.std() or 1)
        method_scores["hybrid"] = alpha * z(method_scores["collaborative"]) + (1 - alpha) * z(method_scores["content"])
        for name, s in method_scores.items():
            s = s.copy()
            s[known] = -np.inf  # never recommend what they already watched
            top = np.argsort(-s)[:k]
            results[name].append(len(set(top) & set(hidden)) / len(hidden))
    return pd.DataFrame({m: {"recall_at_k": float(np.mean(v)) if v else float("nan"), "countries": len(v)}
                         for m, v in results.items()}).T


def because_you_watched(title: str, meta: pd.DataFrame, weeks: pd.DataFrame, n: int = 5,
                        alpha: float = 0.5) -> pd.DataFrame:
    """Hybrid recommendations for one title, with the content and collaborative parts shown separately."""
    items = list(weeks.columns)
    meta = meta.loc[items]
    i = items.index(norm(title))
    content = content_matrix(meta) @ content_matrix(meta)[i]
    cf = _normalise_cols((weeks.values > 0).astype(float))
    collab = cf.T @ cf[:, i]
    z = lambda s: (s - s.mean()) / (s.std() or 1)
    hybrid = alpha * z(collab) + (1 - alpha) * z(content)
    hybrid[i] = -np.inf
    top = np.argsort(-hybrid)[:n]
    return pd.DataFrame({"title": meta["base_title"].iloc[top].values, "genres": meta["genres"].iloc[top].values,
                         "content_sim": content[top].round(2), "audience_overlap": collab[top].round(2)})
