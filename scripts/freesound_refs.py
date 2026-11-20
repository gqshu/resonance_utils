#!/usr/bin/env python3
"""Fetch CC0 reference recordings for each inducer scene from Freesound's public search.

Only Creative Commons 0 sounds of at least 60 s are taken (the licence filter is part of the search
query and re-checked on each sound's page). Downloads are the 128 kbps HQ MP3 previews: originals need
a Freesound login. References are for analysis and SA3 audio-to-audio only; nothing here ships.

    python scripts/freesound_refs.py                    # every scene, 4 per scene
    python scripts/freesound_refs.py rain_soft --per-scene 6 --dry-run

Writes refs/inducer/freesound/<scene>/<id>_<slug>.mp3 and refs/inducer/freesound/credits.json.
"""

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (Macintosh) resonance-refs/0.1"
QUERIES = {
    "train_smooth": ["train interior", "train cabin ambience", "inside train moving", "passenger train interior",
                     "high speed train interior", "intercity train", "train ride ambience"],
    "airplane": ["airplane cabin", "aircraft interior flight"],
    "car_night": ["car interior driving", "car highway interior"],
    "rain_soft": ["rain window", "steady rain", "rain on leaves", "rain on roof", "rain garden", "rain tent"],
    "river": ["wide river", "large river flowing", "river bank ambience", "big river", "river flow calm",
              "river flowing", "stream water", "brook", "creek", "babbling brook", "mountain stream"],
    "ocean_distant": ["ocean waves beach", "distant surf"],
    "wind_pines": ["wind pine trees", "wind forest", "wind in trees", "wind through trees", "breeze leaves",
                   "wind pine forest"],
}
# A result's title must match `need` and not `avoid` (the search also matches tags and descriptions,
# and sorting by downloads lets popular off-topic sounds through).
TITLE = {
    "train_smooth": (r"train|rail|carriage", r"steam|station|horn|whistle|arriv|depart|brake"),
    "airplane": (r"plane|aircraft|cabin|flight|jet", r"take.?off|landing|landed|announce"),
    "car_night": (r"\bcar\b|driv", r"horn|crash|start|door|alarm|race"),
    "rain_soft": (r"rain", r"thunder|storm|drip|umbrella"),
    "river": (r"river|stream|creek|brook|water", r"waterfall|splash|pour|tap|sink"),
    "ocean_distant": (r"wave|ocean|sea|surf|shore|beach", r"storm|gull|boat|harbou?r"),
    "wind_pines": (r"wind|breeze|pine", r"bird|chime|storm|howl|turbine"),
}
ATTR = re.compile(r'data-(sound-id|username|mp3|title|duration|num-downloads)="([^"]*)"')


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def search(q, min_s):
    f = f'license:"Creative Commons 0" duration:[{min_s} TO *]'
    url = "https://freesound.org/search/?" + urllib.parse.urlencode(
        dict(q=q, f=f, s="Downloads (most first)"))
    page = get(url).decode("utf-8", "replace")
    out = []
    for block in page.split('class="bw-player"')[1:]:
        d = dict(ATTR.findall(block[:3000]))
        if "sound-id" in d and "mp3" in d:
            out.append(dict(id=int(d["sound-id"]), user=d["username"], title=html.unescape(d.get("title", "")),
                            duration_s=float(d.get("duration", 0)), downloads=int(d.get("num-downloads", 0)),
                            preview=d["mp3"].replace("-lq.mp3", "-hq.mp3"),
                            url=f"https://freesound.org/people/{d['username']}/sounds/{d['sound-id']}/"))
    return out


def licence_of(sound_url):
    page = get(sound_url).decode("utf-8", "replace")
    m = re.search(r'href="(https?://creativecommons\.org/[^"]+)"', page)
    return m.group(1) if m else None


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", os.path.splitext(s)[0].lower()).strip("-")[:40]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="*", default=list(QUERIES))
    ap.add_argument("--per-scene", type=int, default=5, help="new downloads per scene (kept ones don't count)")
    ap.add_argument("--min-seconds", type=int, default=60)
    ap.add_argument("--out", default="refs/inducer/freesound")
    ap.add_argument("--query", action="append", help="search these instead of the scene's QUERIES (repeatable)")
    ap.add_argument("--avoid", help="title regex to reject instead of the scene's (\"\" keeps all)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cpath = os.path.join(a.out, "credits.json")
    credits = json.load(open(cpath)) if os.path.exists(cpath) else {}
    for sc in a.scenes:
        seen, picks = set(), []
        for q in a.query or QUERIES[sc]:
            for r in search(q, a.min_seconds):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    picks.append(dict(r, query=q))
            time.sleep(1)
        need, avoid = TITLE[sc]
        avoid = avoid if a.avoid is None else (a.avoid or r"(?!)")
        picks = [r for r in picks if re.search(need, r["title"], re.I) and not re.search(avoid, r["title"], re.I)]
        picks.sort(key=lambda r: -r["downloads"])
        print(f"\n{sc}: {len(picks)} CC0 results")
        got = 0
        for r in picks:
            if got >= a.per_scene:
                break
            path = os.path.join(a.out, sc, f"{r['id']}_{slug(r['title'])}.mp3")
            if os.path.exists(path):                 # already have it: re-runs add new ones
                if os.path.relpath(path, a.out) in credits:
                    continue
                os.remove(path)                      # left by a run that died before crediting it
            print(f"  {r['id']:>7d} {r['duration_s']:6.0f} s {r['downloads']:>6d} dl  {r['title'][:50]}  ({r['user']})")
            if a.dry_run:
                got += 1
                continue
            lic = licence_of(r["url"])
            time.sleep(1)
            if not lic or "publicdomain/zero" not in lic:
                print(f"          skipped: licence on the sound page is {lic}")
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(get(r["preview"]))
            time.sleep(1)
            credits[os.path.relpath(path, a.out)] = dict(r, scene=sc, licence=lic)
            with open(cpath, "w") as fh:          # after each one: a dropped connection keeps the credits
                json.dump(credits, fh, indent=2)
            got += 1
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)
        with open(cpath, "w") as fh:
            json.dump(credits, fh, indent=2)
        print(f"\n{len(credits)} references recorded in {cpath}")


if __name__ == "__main__":
    sys.exit(main())
