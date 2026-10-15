"""Load Netflix's official viewing data and join it to IMDb metadata.

Sources (downloaded once into data/raw/, not redistributed):
  - Netflix "What We Watched" report, Jan-Jun 2026: views and hours for every title with >= 100k views
  - Netflix Top 10 (netflix.com/tudum/top10): weekly global lists (with views) and per-country ranks since 2021
  - IMDb non-commercial datasets: genres, year, rating, votes, directors
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
SEASON = re.compile(r":\s*(season|series|limited series|part|volume|vol\.|book|chapter|collection)\b.*$", re.I)


def norm(title: str) -> str:
    """Lowercase, strip accents and punctuation: 'Amélie!' -> 'amelie'."""
    ascii_ = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
    # Non-Latin titles ('धुरंधर', '대홍수') would strip to nothing (or just spaces): keep the original script then.
    t = ascii_ if re.search(r"[A-Za-z]", ascii_) else unicodedata.normalize("NFC", str(title))
    t = re.sub(r"&", " and ", t.lower())
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


YEAR = re.compile(r"\s*\((\d{4})\)\s*$")


def year_hint(title: str) -> int | None:
    """Netflix disambiguates some titles with a year: 'Madagascar (2005)'."""
    m = YEAR.search(str(title).partition(" // ")[0])
    return int(m.group(1)) if m else None


def split_title(title: str) -> tuple[str, str | None, str]:
    """'Bridgerton: Season 4 // Bridgerton: Temporada 4' -> ('Bridgerton', 'Bridgerton', season label)."""
    english, _, original = str(title).partition(" // ")
    english = YEAR.sub("", english)
    season = SEASON.search(english)
    base = SEASON.sub("", english).strip()
    orig_base = original.strip() or None
    if orig_base and season and ": " in orig_base:
        # the original title's season label is in its own language ("Temporada 4", "시즌 3"): cut at its last colon
        orig_base = orig_base.rsplit(": ", 1)[0].strip()
    # Originals in non-Latin scripts are kept: IMDb's originalTitle is often in the same script.
    return base, orig_base, (season.group(0).strip(": ") if season else "")


def engagement(path: Path = RAW / "what_we_watched_2026H1.xlsx") -> pd.DataFrame:
    frames = []
    for sheet, kind in [("Shows", "show"), ("Movies", "movie")]:
        df = pd.read_excel(path, sheet, header=5).iloc[:, 1:]
        df = df[pd.to_numeric(df["Views"], errors="coerce").notna()].copy()
        df.columns = ["title", "global", "release_date", "hours", "runtime", "views"]
        df["kind"] = kind
        frames.append(df)
    e = pd.concat(frames, ignore_index=True)
    e["views"] = e["views"].astype(float)
    e["hours"] = e["hours"].astype(float)
    e["global"] = e["global"].eq("Yes")
    e["release_date"] = pd.to_datetime(e["release_date"], errors="coerce")
    # Netflix gives a release date only for titles it released itself (originals and exclusive releases).
    e["netflix_release"] = e["release_date"].notna()
    parts = e["title"].map(split_title)
    e["base_title"] = parts.str[0]
    e["original_title"] = parts.str[1]
    e["season"] = parts.str[2]
    e["non_english"] = e["title"].str.contains(" // ")
    e["year_hint"] = e["title"].map(year_hint)
    return e


def top10_global(path: Path = RAW / "all-weeks-global.tsv") -> pd.DataFrame:
    g = pd.read_csv(path, sep="\t", parse_dates=["week"])
    g["kind"] = g["category"].str.startswith("TV").map({True: "show", False: "movie"})
    g["english"] = g["category"].str.contains("English", case=False) & ~g["category"].str.contains("Non-English")
    return g


def top10_countries(path: Path = RAW / "all-weeks-countries.tsv") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", parse_dates=["week"])


def most_popular(path: Path = RAW / "most-popular.tsv") -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


# --- IMDb ------------------------------------------------------------------------------------------

MOVIE_TYPES = {"movie", "tvMovie", "tvSpecial", "video"}
SHOW_TYPES = {"tvSeries", "tvMiniSeries"}


def imdb_titles(raw: Path = RAW) -> pd.DataFrame:
    """Movies and series with their rating, votes and genres (a few hundred thousand rows after filtering)."""
    usecols = ["tconst", "titleType", "primaryTitle", "originalTitle", "startYear", "genres"]
    basics = pd.read_csv(raw / "title.basics.tsv.gz", sep="\t", usecols=usecols, na_values="\\N",
                         dtype=str, quoting=3)
    basics = basics[basics["titleType"].isin(MOVIE_TYPES | SHOW_TYPES)]
    ratings = pd.read_csv(raw / "title.ratings.tsv.gz", sep="\t", na_values="\\N")
    t = basics.merge(ratings, on="tconst", how="inner")  # only titles people have rated
    t["startYear"] = pd.to_numeric(t["startYear"], errors="coerce")
    t["kind"] = t["titleType"].map(lambda x: "show" if x in SHOW_TYPES else "movie")
    return t


def match_imdb(e: pd.DataFrame, imdb: pd.DataFrame, min_votes: int = 50) -> pd.DataFrame:
    """Attach the best IMDb match to each Netflix title.

    Candidates share the normalised title (English or original name) and kind (movie/show). When Netflix gives a
    release date, the IMDb year must be within one year of it; otherwise the most-voted candidate wins.
    """
    imdb = imdb[imdb["numVotes"] >= min_votes]
    keys = pd.concat([
        imdb.assign(key=imdb["primaryTitle"].map(norm)),
        imdb.assign(key=imdb["originalTitle"].map(norm)),
    ]).drop_duplicates(["tconst", "key"])
    keys = keys[keys["key"] != ""]  # titles like '$' normalise to nothing and must never match
    by_key = {k: g for k, g in keys.groupby(["key", "kind"])}

    rows = []
    for i, r in e.iterrows():
        names = [r["base_title"], r["original_title"]]
        if r["kind"] == "show":
            # 'Stranger Things 5': a bare trailing number is a season number
            names.append(re.sub(r"\s+\d{1,2}$", "", r["base_title"]))
        cands = []
        for name in dict.fromkeys(n for n in names if n):
            if not norm(name):
                continue
            g = by_key.get((norm(name), r["kind"]))
            if g is not None:
                cands.append(g)
        if not cands:
            continue
        c = pd.concat(cands).drop_duplicates("tconst")
        order = ["numVotes"]
        if pd.notna(r["year_hint"]):
            c = c[(c["startYear"] - r["year_hint"]).abs() <= 1]
        elif pd.notna(r["release_date"]):
            y = r["release_date"].year
            if r["kind"] == "movie":
                c = c[(c["startYear"] - y).abs() <= 1]
            else:
                # A season can come years after the series began, so take the latest series that had started by
                # then: Netflix's 2023 'One Piece' rather than the 1999 anime.
                c = c[c["startYear"] <= y + 1]
                order = ["startYear", "numVotes"]
        if c.empty:
            continue
        best = c.sort_values(order, ascending=False).iloc[0]
        rows.append({"idx": i, "tconst": best["tconst"], "imdb_title": best["primaryTitle"],
                     "imdb_year": best["startYear"], "rating": best["averageRating"], "votes": best["numVotes"],
                     "genres": best["genres"], "n_candidates": len(c)})
    m = pd.DataFrame(rows).set_index("idx")
    return e.join(m)


def directors(tconsts: set[str], raw: Path = RAW) -> pd.DataFrame:
    """(tconst, director name) pairs for the given titles. Reads the large files in chunks and keeps only matches."""
    crew = []
    for chunk in pd.read_csv(raw / "title.crew.tsv.gz", sep="\t", usecols=["tconst", "directors"], na_values="\\N",
                             dtype=str, chunksize=500_000, quoting=3):
        crew.append(chunk[chunk["tconst"].isin(tconsts)].dropna())
    crew = pd.concat(crew)
    crew = crew.assign(nconst=crew["directors"].str.split(",")).explode("nconst")[["tconst", "nconst"]]
    wanted = set(crew["nconst"])
    names = []
    for chunk in pd.read_csv(raw / "name.basics.tsv.gz", sep="\t", usecols=["nconst", "primaryName"], dtype=str,
                             chunksize=1_000_000, quoting=3):
        names.append(chunk[chunk["nconst"].isin(wanted)])
    return crew.merge(pd.concat(names), on="nconst", how="left").rename(columns={"primaryName": "director"})
