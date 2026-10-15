#!/bin/sh
# Download the source files into data/raw/ (about 600 MB). Netflix files are published by Netflix for download;
# IMDb datasets are for personal and non-commercial use (https://developer.imdb.com/non-commercial-datasets/).
set -e
mkdir -p data/raw && cd data/raw
UA="Mozilla/5.0 (compatible; netflix-insights-study/0.1; personal non-commercial)"
for f in all-weeks-global.tsv all-weeks-countries.tsv most-popular.tsv; do
  curl -sfL -A "$UA" -o "$f" "https://www.netflix.com/tudum/top10/data/$f"; sleep 2
done
# Netflix "What We Watched" report, Jan-Jun 2026 (linked from about.netflix.com/news/what-we-watched-the-first-half-of-2026)
curl -sfL -A "$UA" -o what_we_watched_2026H1.xlsx \
  "https://assets.ctfassets.net/4cd45et68cgf/40WGcHJa9vRua31kU6Gbz5/43b15c7fbd6924392fc80e885839e39f/Netflix-s_What_We_Watched_Report_2026Jan-Jun__1_.xlsx"
for f in title.basics title.ratings title.crew name.basics; do
  curl -sfL -A "$UA" -o "$f.tsv.gz" "https://datasets.imdbws.com/$f.tsv.gz"
done
ls -la
