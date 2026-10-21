"""Fetch openly licensed ads and advertising posters from Wikimedia Commons for `bench`.

    python benchmarks/fetch_wikimedia_ads.py --limit 300

Writes $MA_BENCH/data/real/wikimedia/<file>.jpg plus manifest.csv (title, licence, author, source).
Only public-domain, CC0, CC BY and CC BY-SA files are kept; images are 1024 px thumbnails.
"""
import argparse
import csv
import os
import re
import time
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "MediaAttention-research/0.1 (https://github.com/pranikz/MediaAttention)"}
CATEGORIES = ["Advertisements", "Advertising posters", "Food and drink advertising posters",
              "Aviation advertising posters", "Bicycle advertising posters", "Unidentified advertisements"]
SKIP = re.compile(r"enslaved|slave", re.I)  # historical slave-sale notices are not ads to benchmark
OPEN = re.compile(r"^(public domain|pd|cc0|cc by(-sa)?\b)", re.I)


def files_in(category):
    params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}", "cmtype": "file",
              "cmlimit": 500, "format": "json"}
    while True:
        d = requests.get(API, params=params, headers=UA, timeout=30).json()
        yield from (m["title"] for m in d["query"]["categorymembers"])
        if "continue" not in d:
            return
        params.update(d["continue"])


def info(titles):
    d = requests.get(API, headers=UA, timeout=30, params={
        "action": "query", "titles": "|".join(titles), "prop": "imageinfo", "format": "json",
        "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 1024}).json()
    for page in d["query"]["pages"].values():
        if "imageinfo" in page:
            yield page["title"], page["imageinfo"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    a = ap.parse_args()
    out = Path(os.environ.get("MA_BENCH", "~/Documents/Personal/MediaAttention-bench")).expanduser() / "data/real/wikimedia"
    out.mkdir(parents=True, exist_ok=True)
    titles = list(dict.fromkeys(t for c in CATEGORIES for t in files_in(c) if not SKIP.search(t)))
    print(f"{len(titles)} candidate files")
    rows = []
    for i in range(0, len(titles), 50):
        for title, ii in info(titles[i:i + 50]):
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if ii["mime"] not in ("image/jpeg", "image/png") or min(ii["width"], ii["height"]) < 400 or not OPEN.match(lic):
                continue
            name = re.sub(r"[^A-Za-z0-9._-]+", "_", title[5:])[:120].rsplit(".", 1)[0] + ".jpg"
            r = requests.get(ii.get("thumburl", ii["url"]), headers=UA, timeout=60)
            if r.ok:
                (out / name).write_bytes(r.content)
                artist = re.sub("<[^>]+>", "", meta.get("Artist", {}).get("value", "")).strip()
                rows.append({"file": name, "title": title, "licence": lic, "author": artist,
                             "source": ii["descriptionurl"]})
                print(f"[{len(rows)}] {name} ({lic})")
            time.sleep(0.2)  # be gentle with Commons
            if len(rows) >= a.limit:
                break
        if len(rows) >= a.limit:
            break
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "title", "licence", "author", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} images to {out}")


if __name__ == "__main__":
    main()
