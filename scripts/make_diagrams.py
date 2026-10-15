"""Draw docs/how-it-works.svg and docs/what-we-get.svg from data/findings.json.

    python scripts/make_diagrams.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diagram_kit import BODY, INK, MUTED, arrow, box, card, hbar, lane, lines, panel, svg, t  # noqa: E402

F = json.loads(Path("data/findings.json").read_text())
D, C, R, DEC = F["data"], F["concentration"], F["recommenders"], F["decay"]

b = [lane(24, 104, 1152, 236, "blue", "DATA · official downloads, no scraping")]
b.append(box(44, 144, 270, 160, "slate", "Netflix Top 10",
             ["weekly global lists with views", f"+ Top 10 ranks in {D['countries']} countries", "since July 2021"], plain=True))
b.append(box(334, 144, 270, 160, "slate", "What We Watched report",
             [f"{D['titles']:,} titles, {D['views_billion']}B views", "Jan-Jun 2026, every title", "with 100k+ views"], plain=True))
b.append(box(624, 144, 250, 160, "slate", "IMDb datasets",
             ["genres, year, rating,", "votes, directors", "(non-commercial use)"], plain=True))
b.append(box(910, 144, 246, 160, "green", "Title matching",
             [f"{D['matched_view_pct']:.0f}% of views matched", "season and year rules,", "English + original titles", "random spot-checks: correct"]))
b.append(arrow(874, 224, 906, 224))
b.append(lane(24, 360, 1152, 280, "violet", "ANALYSE · pandas + numpy"))
b.append(box(44, 400, 260, 210, "violet", "What the world watches",
             ["top films and series", f"top 1% of titles get {C['top_1pct_share']*100:.0f}%", "of all views", "genres, kids & family", "non-English share"]))
b.append(box(324, 400, 260, 210, "violet", "Directors",
             ["films only (series credit", "every episode director)", "best: IMDb rating, 3+", "films, 20k+ votes each", "most watched: total views"]))
b.append(box(604, 400, 260, 210, "violet", "Time and place",
             ["weekly decay after debut", "release weekday", "longest in the Top 10", "country taste overlap", "local hits (India)"]))
b.append(box(884, 400, 272, 210, "blue", "Recommenders, tested",
             ["content: genre, director,", "decade, language", "collaborative: same audiences", f"({D['countries']} countries as 'viewers')", "hide 20%, measure recall@20", "vs a popularity baseline"]))
Path("docs/how-it-works.svg").write_text(svg(1200, 660, "Netflix insights: how it works",
    "What the world watches, using only data Netflix and IMDb publish for download", "\n".join(b)))

b = [panel(24, 104, 560, 330, "Best vs most watched film directors", "Jan-Jun 2026 views · IMDb ratings")]
best = list(F["best_directors"].items())[:3]
most = list(F["most_watched_directors"].items())[:3]
b.append(t(44, 180, "Highest rated (average IMDb)", 12, MUTED))
for i, (n, v) in enumerate(best):
    b.append(t(44, 206 + i * 24, n, 14, INK, 600))
    b.append(t(540, 206 + i * 24, f"{v['avg_rating']:.2f}", 14, "#1D4ED8", 700, "end"))
b.append(t(44, 296, "Most watched (total views)", 12, MUTED))
for i, (n, v) in enumerate(most):
    b.append(t(44, 322 + i * 24, f"{n} ({v['most_watched_title']})", 14, INK, 600))
    b.append(t(540, 322 + i * 24, f"{v['total_views'] / 1e6:.0f}M", 14, "#B45309", 700, "end"))
b.append(t(44, 414, "No overlap: viewing follows new releases, not acclaim.", 12, BODY))
b.append(panel(608, 104, 568, 330, "Which recommender predicts viewing?", f"recall@20 on hidden titles, {int(R['collaborative']['countries'])} countries"))
for i, (lab, key, col) in enumerate([("Content", "content", "#94A3B8"), ("Most popular", "popularity", "#94A3B8"),
                                     ("Collaborative", "collaborative", "#2563EB")]):
    v = R[key]["recall_at_k"] * 100
    b.append(hbar(618, 170 + i * 44, lab, v, 20, 230, col, f"{v:.1f}%", label_w=150, bold=key == "collaborative"))
b.append(lines(628, 334, ["Audience behaviour predicts viewing; genre", "similarity alone barely does. Content still",
                          "helps with brand-new titles nobody has seen."], 13, BODY, 22))
b.append(card(24, 458, 276, 182, "blue", f"{C['top_1pct_share'] * 100:.0f}%", "of views go to the top 1%",
              [f"top 10% of titles: {C['top_10pct_share'] * 100:.0f}% of views;", f"Netflix releases: {C['netflix_release_title_share'] * 100:.0f}% of titles,",
               f"{C['netflix_release_share'] * 100:.0f}% of views"]))
b.append(card(316, 458, 276, 182, "amber", f"{DEC['4']['movie'] * 100:.0f}%", "of peak by week 4 (films)",
              [f"series still hold {DEC['4']['show'] * 100:.0f}%; most", "titles peak in their first", "chart week"]))
tw = F["taste_twins"]
b.append(card(608, 458, 276, 182, "green", f"{tw['Germany']['jaccard']:.2f}", "Germany-Austria taste overlap",
              [f"India's twin: {tw['India']['most_similar']} ({tw['India']['jaccard']:.2f})", "Japan is the most distinct:",
               f"best match only {tw['Japan']['jaccard']:.2f}"]))
lr = F["longest_running"][0]
b.append(card(900, 458, 276, 182, "purple", f"{lr['cumulative_weeks_in_top_10']} wks", lr["show_title"],
              ["longest run in the global", f"Top 10; {F['titles_in_every_country']} titles charted in", f"all {F['countries_considered']} countries this year"]))
Path("docs/what-we-get.svg").write_text(svg(1200, 660, "Netflix insights: what we get from it",
    f"{D['titles']:,} titles · {D['views_billion']}B views (Jan-Jun 2026) · Top 10 in {D['countries']} countries since 2021", "\n".join(b)))
print("diagrams written")
