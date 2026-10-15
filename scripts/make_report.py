"""Build the matched dataset, compute every finding, and draw the charts.

    PYTHONPATH=src python scripts/make_report.py

Needs the files in data/raw/ (see scripts/download.sh). Writes data/*.pkl and data/findings.json (local only)
and docs/images/*.png (aggregate charts, safe to publish).
"""

import json
from pathlib import Path

import pandas as pd

from nfx import analysis as A
from nfx import charts as C
from nfx import load as L
from nfx import recommend as R

IMG = Path("docs/images")
IMG.mkdir(parents=True, exist_ok=True)
SINCE = "2025-10-01"  # "the last year" of country charts

e = L.engagement()
m = L.match_imdb(e, L.imdb_titles())
crew = L.directors(set(m["tconst"].dropna()))
g, c = L.top10_global(), L.top10_countries()
m.to_pickle("data/matched.pkl")
crew.to_pickle("data/crew.pkl")

matched = m["tconst"].notna()
conc = A.concentration(m)
movies, shows = A.top_titles(m, "movie"), A.top_titles(m, "show")
genres = A.by_genre(m)
kids = A.kids_share(m)
rv = A.rating_vs_views(m)
d = A.director_table(m, crew)
best, most = A.best_directors(d, min_titles=3, min_votes=20000), A.most_watched_directors(d)
dec = A.weekly_decay(g)
peak = A.peak_week(g)
wd = A.release_weekday(m)
longest = A.longest_running(g)
glob = A.most_global_titles(c, SINCE, n=200)
twins = A.taste_twins(c, SINCE)
india_local = A.local_hits(c, "India", SINCE)
meta, weeks = R.build_items(m, crew, c, since="2025-01-01")
ev = pd.concat([R.evaluate(meta, weeks, seed=s) for s in (1, 2, 3)]).groupby(level=0).mean()
examples = {t: R.because_you_watched(t, meta, weeks, alpha=0.9).to_dict("records")
            for t in ["Stranger Things", "KPop Demon Hunters", "Squid Game", "Bridgerton"]}


def short(t):
    t = t.split(" // ")[0]
    return t if len(t) <= 38 else t[:36] + "..."


C.save(C.hbars([short(t) for t in movies["title"]], (movies["views"] / 1e6).tolist(), "Most watched films, Jan-Jun 2026",
               "views (millions)", "{:.0f}M"), IMG / "top_movies.png")
C.save(C.hbars([short(t) for t in shows["title"]], (shows["views"] / 1e6).tolist(), "Most watched series, Jan-Jun 2026",
               "views (millions)", "{:.0f}M"), IMG / "top_shows.png")
gt = genres.head(12)
C.save(C.hbars(gt.index.tolist(), (gt["share_of_views"] * 100).tolist(), "Share of all views by genre",
               "% of views (a title's views are split across its genres)", "{:.1f}%"), IMG / "genres.png")
C.save(C.hbars([f"{n} ({r.titles} films)" for n, r in best.iterrows()], best["avg_rating"].tolist(),
               "Best film directors on Netflix", "average IMDb rating of their films on Netflix", "{:.2f}",
               note="3+ films on Netflix, each with 20,000+ IMDb votes"), IMG / "best_directors.png")
C.save(C.hbars([f"{n} ({r.most_watched_title})" for n, r in most.iterrows()], (most["total_views"] / 1e6).tolist(),
               "Most watched film directors, Jan-Jun 2026", "views of their films (millions)", "{:.0f}M",
               color="#eb6834"), IMG / "most_watched_directors.png")
C.save(C.decay(dec), IMG / "decay.png")
C.save(C.weekday(wd), IMG / "release_weekday.png")
top_rating_lift = rv.loc["8+", "median_views"] / rv.drop("8+")["median_views"].median()
C.save(C.rating_views(rv, f"Ratings barely matter, except at 8+ ({top_rating_lift:.1f}x the median views)"),
       IMG / "rating_vs_views.png")
focus = ["India", "Pakistan", "Bangladesh", "United States", "United Kingdom", "Germany", "Austria", "Brazil",
         "Mexico", "Japan", "South Korea", "Taiwan", "Nigeria", "Kenya"]
jm = A.jaccard_matrix({k: v for k, v in A.country_sets(c, SINCE).items() if k in focus}).loc[focus, focus]
C.save(C.heatmap(jm, "Which countries watch the same things? (last 12 months)"), IMG / "country_taste.png")
C.save(C.recommenders(ev), IMG / "recommenders.png")

in_all = (glob == 1.0).sum()
findings = {
    "data": {"titles": int(len(m)), "views_billion": round(m["views"].sum() / 1e9, 1),
             "matched_title_pct": round(100 * matched.mean(), 1),
             "matched_view_pct": round(100 * m.loc[matched, "views"].sum() / m["views"].sum(), 1),
             "countries": int(c["country_name"].nunique()), "global_weeks": [str(g["week"].min().date()), str(g["week"].max().date())]},
    "concentration": {k: (round(v, 3) if isinstance(v, float) else int(v)) for k, v in conc.items()},
    "top_movies": movies[["title", "views", "rating"]].to_dict("records"),
    "top_shows": shows[["title", "views", "rating"]].to_dict("records"),
    "genres": genres.head(12)[["titles", "share_of_views", "median_views_per_title", "avg_rating"]].round(3).to_dict("index"),
    "kids_family": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in kids.items()},
    "rating_vs_views": rv.to_dict("index"),
    "best_directors": best[["titles", "avg_rating", "total_views", "most_watched_title"]].round(2).to_dict("index"),
    "most_watched_directors": most[["titles", "avg_rating", "total_views", "most_watched_title"]].round(2).to_dict("index"),
    "decay": dec.round(3).to_dict("index"),
    "peak_week_share": {int(k): round(v, 3) for k, v in peak.items()},
    "release_weekday": wd.round(3).to_dict("index"),
    "longest_running": longest.to_dict("records"),
    "titles_in_every_country": int(in_all), "countries_considered": int(c[c["week"] >= SINCE]["country_name"].nunique()),
    "taste_twins": twins.round(2).to_dict("index"),
    "india_local_hits": india_local.reset_index().to_dict("records"),
    "recommenders": ev.round(4).to_dict("index"),
    "recommender_items": list(weeks.shape),
    "because_you_watched": examples,
}
Path("data/findings.json").write_text(json.dumps(findings, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: findings[k] for k in ["data", "concentration", "kids_family", "recommenders",
                                           "titles_in_every_country", "peak_week_share"]}, indent=1, default=str))
