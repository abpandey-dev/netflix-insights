"""What the official numbers say. Every function takes tidy DataFrames and returns a small summary table."""

from __future__ import annotations

import numpy as np
import pandas as pd

KIDS_GENRES = {"Animation", "Family"}


# --- 1. What the world watches --------------------------------------------------------------------

def top_titles(e: pd.DataFrame, kind: str, n: int = 10) -> pd.DataFrame:
    t = e[e["kind"] == kind].nlargest(n, "views")
    return t[["title", "views", "hours", "release_date", "rating", "genres"]]


def concentration(e: pd.DataFrame) -> dict:
    v = e["views"].sort_values(ascending=False)
    total = v.sum()
    n = len(v)
    return {
        "titles": n,
        "total_views": total,
        "top_1pct_share": v.head(max(1, n // 100)).sum() / total,
        "top_10pct_share": v.head(n // 10).sum() / total,
        "netflix_release_share": e.loc[e["netflix_release"], "views"].sum() / total,
        "netflix_release_title_share": e["netflix_release"].mean(),
        "non_english_share": e.loc[e["non_english"], "views"].sum() / total,
        "movie_share": e.loc[e["kind"] == "movie", "views"].sum() / total,
    }


# --- 2. Genres and audience -----------------------------------------------------------------------

def genres_long(e: pd.DataFrame) -> pd.DataFrame:
    """One row per (title, genre). Views are split evenly across a title's genres so totals still add up."""
    g = e.dropna(subset=["genres"]).copy()
    g["genre"] = g["genres"].str.split(",")
    g["n_genres"] = g["genre"].str.len()
    g = g.explode("genre")
    g["views_share"] = g["views"] / g["n_genres"]
    return g


def by_genre(e: pd.DataFrame, min_titles: int = 30) -> pd.DataFrame:
    g = genres_long(e)
    out = g.groupby("genre").agg(titles=("title", "size"), views=("views_share", "sum"),
                                 median_views_per_title=("views", "median"), avg_rating=("rating", "mean"))
    out["share_of_views"] = out["views"] / g["views_share"].sum()
    return out[out["titles"] >= min_titles].sort_values("views", ascending=False)


def kids_share(e: pd.DataFrame) -> dict:
    """Netflix publishes no viewer ages. The closest honest proxy: titles IMDb tags Animation or Family."""
    m = e.dropna(subset=["genres"])
    kids = m["genres"].str.split(",").map(lambda gs: bool(KIDS_GENRES & set(gs)))
    top = m[kids].nlargest(5, "views")["title"].tolist()
    return {"kids_family_view_share": m.loc[kids, "views"].sum() / m["views"].sum(),
            "kids_family_title_share": kids.mean(), "top_kids_family": top}


def rating_vs_views(e: pd.DataFrame) -> pd.DataFrame:
    """Do better-rated titles get watched more? Views by IMDb rating band (titles with 1,000+ votes)."""
    m = e[(e["votes"] >= 1000)].copy()
    m["band"] = pd.cut(m["rating"], [0, 5, 6, 7, 8, 10], labels=["<5", "5-6", "6-7", "7-8", "8+"])
    return m.groupby("band", observed=True)["views"].agg(titles="size", median_views="median", mean_views="mean")


# --- 3/4. Directors ---------------------------------------------------------------------------------

def director_table(e: pd.DataFrame, crew: pd.DataFrame) -> pd.DataFrame:
    """Per film director: films on Netflix, total views, average IMDb rating.

    Films only: for a series IMDb credits every episode director, so "the director of Bridgerton" isn't one person.
    """
    m = e[(e["kind"] == "movie")].dropna(subset=["tconst"]).merge(crew[["tconst", "director"]], on="tconst")
    per_title = m.groupby(["director", "tconst"]).agg(views=("views", "sum"), rating=("rating", "first"),
                                                      votes=("votes", "first"), title=("imdb_title", "first"))
    d = per_title.groupby("director").agg(titles=("title", "size"), total_views=("views", "sum"),
                                          avg_rating=("rating", "mean"), min_votes=("votes", "min"))
    top_title = per_title.reset_index().sort_values("views").groupby("director").tail(1).set_index("director")["title"]
    d["most_watched_title"] = top_title
    return d


def best_directors(d: pd.DataFrame, min_titles: int = 3, min_votes: int = 5000, n: int = 10) -> pd.DataFrame:
    """Highest average IMDb rating among directors with several well-known titles on Netflix."""
    q = d[(d["titles"] >= min_titles) & (d["min_votes"] >= min_votes)]
    return q.sort_values(["avg_rating", "total_views"], ascending=False).head(n)


def most_watched_directors(d: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return d.sort_values("total_views", ascending=False).head(n)


# --- Time: how interest rises and fades ------------------------------------------------------------

def weekly_decay(g: pd.DataFrame, max_week: int = 8) -> pd.DataFrame:
    """For titles whose first Top 10 week is in the data, median views in week N as a share of their best week."""
    v = g.dropna(subset=["weekly_views"]).copy()
    v["key"] = v["show_title"] + " | " + v["season_title"].fillna("")
    v = v.sort_values("week")
    first = v.groupby("key")["week"].transform("min")
    v["week_n"] = ((v["week"] - first).dt.days // 7) + 1
    peak = v.groupby("key")["weekly_views"].transform("max")
    v["of_peak"] = v["weekly_views"] / peak
    # only titles that debuted inside the data (their first chart week is a real debut: cumulative weeks == 1)
    debut_keys = v.loc[(v["week_n"] == 1) & (v["cumulative_weeks_in_top_10"] == 1), "key"]
    v = v[v["key"].isin(set(debut_keys)) & (v["week_n"] <= max_week)]
    out = v.groupby(["week_n", "kind"])["of_peak"].median().unstack()
    out["titles_in_week"] = v.groupby("week_n")["key"].nunique()
    return out


def peak_week(g: pd.DataFrame) -> pd.Series:
    """Which chart week (1st, 2nd, ...) is a title's biggest? Share of debuting titles."""
    v = g.dropna(subset=["weekly_views"]).copy()
    v["key"] = v["show_title"] + " | " + v["season_title"].fillna("")
    first = v.groupby("key")["week"].transform("min")
    v["week_n"] = ((v["week"] - first).dt.days // 7) + 1
    debut = set(v.loc[(v["week_n"] == 1) & (v["cumulative_weeks_in_top_10"] == 1), "key"])
    v = v[v["key"].isin(debut)]
    best = v.loc[v.groupby("key")["weekly_views"].idxmax(), "week_n"]
    return best.clip(upper=4).value_counts(normalize=True).sort_index()


def release_weekday(e: pd.DataFrame) -> pd.DataFrame:
    """Netflix's own releases in the report: which weekday do they come out, and how do they do?"""
    r = e[e["netflix_release"] & (e["release_date"] >= "2026-01-01")].copy()
    r["weekday"] = r["release_date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    out = r.groupby("weekday")["views"].agg(releases="size", median_views="median").reindex(order).fillna(0)
    out["share_of_releases"] = out["releases"] / out["releases"].sum()
    return out


def longest_running(g: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t = g.groupby(["show_title", "season_title", "category"], dropna=False)["cumulative_weeks_in_top_10"].max()
    return t.sort_values(ascending=False).head(n).reset_index()


# --- Countries -------------------------------------------------------------------------------------

def country_sets(c: pd.DataFrame, since: str = "2025-10-01") -> dict[str, set]:
    r = c[c["week"] >= since]
    return {k: set(v) for k, v in r.groupby("country_name")["show_title"]}


def most_global_titles(c: pd.DataFrame, since: str = "2025-10-01", n: int = 10) -> pd.Series:
    r = c[c["week"] >= since]
    total = r["country_name"].nunique()
    return (r.groupby("show_title")["country_name"].nunique() / total).sort_values(ascending=False).head(n)


def taste_twins(c: pd.DataFrame, since: str = "2025-10-01") -> pd.DataFrame:
    """Jaccard overlap of each country's Top 10 titles over the last year: who watches like whom?"""
    sets = country_sets(c, since)
    names = sorted(sets)
    rows = []
    for a in names:
        best, score = None, -1.0
        for b in names:
            if a == b:
                continue
            j = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            if j > score:
                best, score = b, j
        rows.append({"country": a, "most_similar": best, "jaccard": score, "titles": len(sets[a])})
    return pd.DataFrame(rows).set_index("country")


def local_hits(c: pd.DataFrame, country: str, since: str = "2025-10-01", max_countries: int = 3,
               n: int = 10) -> pd.DataFrame:
    """Titles that were big in one country but charted in almost no others."""
    r = c[c["week"] >= since]
    reach = r.groupby("show_title")["country_name"].nunique()
    mine = r[r["country_name"] == country].groupby("show_title").agg(weeks=("week", "nunique"),
                                                                    best_rank=("weekly_rank", "min"))
    mine["countries"] = reach.reindex(mine.index)
    return mine[mine["countries"] <= max_countries].sort_values(["weeks", "best_rank"], ascending=[False, True]).head(n)


def jaccard_matrix(sets: dict[str, set]) -> pd.DataFrame:
    names = sorted(sets)
    m = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            m[i, j] = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    return pd.DataFrame(m, index=names, columns=names)
