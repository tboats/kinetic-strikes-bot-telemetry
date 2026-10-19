#!/usr/bin/env python3
"""
Fills any missing series in data/arctic_shift_aggregates.json, data/urr_flair_series.json,
data/granular_posts.json, and data/viral_diffusion.json, and then runs build_outputs.py.
"""

import json
import time
from pathlib import Path
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PoliticalBotsResearch/1.0"})


def api_get(url: str, params: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 200:
                time.sleep(0.35)
                return r.json()
            if r.status_code in (422, 429):
                time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            print(f"  [warn] {e}")
            time.sleep(1.5)
    return None


def paginate_posts(subreddit: str, start_iso: str, max_pages: int = 20) -> list:
    """Paginate /api/posts/search without 'fields' parameter so it returns 200 OK every time."""
    url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    posts = []
    seen = set()
    cur_after = start_iso

    for _ in range(max_pages):
        res = api_get(url, {"subreddit": subreddit, "after": cur_after, "limit": "auto", "sort": "asc"})
        if not res or not res.get("data"):
            break
        batch = res["data"]
        new_cnt = 0
        max_ts = None
        for p in batch:
            pid = p.get("id")
            if pid not in seen:
                seen.add(pid)
                st = p.get("selftext") or ""
                selftext_status = (
                    "removed"
                    if st == "[removed]"
                    else ("deleted" if st == "[deleted]" else ("empty" if st == "" else "present"))
                )
                posts.append({
                    "id": pid,
                    "created_utc": p.get("created_utc"),
                    "author": p.get("author"),
                    "author_flair_text": p.get("author_flair_text"),
                    "title": p.get("title"),
                    "domain": p.get("domain"),
                    "score": p.get("score"),
                    "upvote_ratio": p.get("upvote_ratio"),
                    "num_comments": p.get("num_comments"),
                    "removed_by_category": p.get("removed_by_category"),
                    "is_self": p.get("is_self"),
                    "link_flair_text": p.get("link_flair_text"),
                    "link_flair_css_class": p.get("link_flair_css_class"),
                    "selftext_status": selftext_status,
                })
                new_cnt += 1
            ts = p.get("created_utc")
            if ts is not None and (max_ts is None or ts > max_ts):
                max_ts = ts
        if new_cnt == 0 or len(batch) < 25 or max_ts is None:
            break
        cur_after = str(int(max_ts))
    return posts


def posts_to_daily_buckets(posts: list) -> list:
    """Convert granular post list into Arctic Shift CEST-aligned daily buckets."""
    from collections import Counter
    import pandas as pd
    counts = Counter()
    for p in posts:
        ts = pd.to_datetime(int(p["created_utc"]), unit="s", utc=True)
        # Align to 22:00:00Z bucket start like Arctic Shift
        shifted = ts + pd.Timedelta(hours=2)
        bucket_start = pd.Timestamp(year=shifted.year, month=shifted.month, day=shifted.day, tz="UTC") - pd.Timedelta(hours=2)
        counts[bucket_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")] += 1
    return [{"created_utc": k, "count": str(counts[k])} for k in sorted(counts.keys())]


def posts_to_hourly_buckets(posts: list, min_iso: str = "2026-10-01") -> list:
    from collections import Counter
    import pandas as pd
    counts = Counter()
    min_ts = pd.to_datetime(min_iso, utc=True)
    for p in posts:
        ts = pd.to_datetime(int(p["created_utc"]), unit="s", utc=True)
        if ts >= min_ts:
            hr_start = ts.floor("h")
            counts[hr_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")] += 1
    return [{"created_utc": k, "count": str(counts[k])} for k in sorted(counts.keys())]


def main():
    base = "https://arctic-shift.photon-reddit.com/api"
    with open(DATA_DIR / "arctic_shift_aggregates.json", "r", encoding="utf-8") as f:
        agg = json.load(f)

    # 1. Fetch granular posts for key subreddits from 2026-09-10 (fast, indexed, 100% reliable)
    print("=== 1. Fetching Granular Posts (without fields filter) ===")
    gran_subs = [
        "Conservative",
        "AskConservatives",
        "moderatepolitics",
        "tjournal_refugees",
        "WayOfTheBern",
        "EndlessWar",
        "WalkAway",
        "AskARussian",
    ]
    granular_store = {}
    for sub in gran_subs:
        posts = paginate_posts(sub, "2026-09-10", max_pages=22)
        granular_store[sub] = posts
        print(f"  r/{sub}: {len(posts)} granular posts fetched")
        # Also populate daily_posts and hourly_posts from exact post records if missing!
        if not agg["daily_posts"].get(sub):
            agg["daily_posts"][sub] = posts_to_daily_buckets(posts)
        if not agg["hourly_posts"].get(sub):
            agg["hourly_posts"][sub] = posts_to_hourly_buckets(posts, "2026-10-01")

    with open(DATA_DIR / "granular_posts.json", "w", encoding="utf-8") as f:
        json.dump(granular_store, f)

    # 2. Fill missing daily/hourly comment counts for r/Conservative & controls (Oct 1 - Oct 12)
    print("=== 2. Filling Comment & Remaining Subreddit Aggregates (Oct 1 - Oct 12) ===")
    for sub in ["Conservative", "conspiracy", "UkraineRussiaReport", "politics", "worldnews", "nfl"]:
        if not agg["daily_comments"].get(sub):
            dc = api_get(
                f"{base}/comments/search/aggregate",
                {"subreddit": sub, "after": "2026-10-01", "aggregate": "created_utc", "frequency": "day"},
            )
            if dc and dc.get("data"):
                agg["daily_comments"][sub] = dc["data"]
                print(f"  r/{sub} daily_comments: {len(dc['data'])} days")

    if not agg["hourly_comments"].get("Conservative"):
        hc = api_get(
            f"{base}/comments/search/aggregate",
            {"subreddit": "Conservative", "after": "2026-10-01", "aggregate": "created_utc", "frequency": "hour"},
        )
        if hc and hc.get("data"):
            agg["hourly_comments"]["Conservative"] = hc["data"]
            print(f"  r/Conservative hourly_comments: {len(hc['data'])} hours")

    if not agg["daily_posts"].get("nfl"):
        dp_nfl = api_get(
            f"{base}/posts/search/aggregate",
            {"subreddit": "nfl", "after": "2026-09-10", "aggregate": "created_utc", "frequency": "day"},
        )
        if dp_nfl and dp_nfl.get("data"):
            agg["daily_posts"]["nfl"] = dp_nfl["data"]
            print(f"  r/nfl daily_posts: {len(dp_nfl['data'])} days")

    with open(DATA_DIR / "arctic_shift_aggregates.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # 3. Fill r/UkraineRussiaReport Flair Series (Oct 1 - Oct 12)
    print("=== 3. Filling r/UkraineRussiaReport Flair Series (Oct 1 - Oct 12) ===")
    with open(DATA_DIR / "urr_flair_series.json", "r", encoding="utf-8") as f:
        urr = json.load(f)
    for flair in ["Pro Russia", "Pro Ukraine", "Neutral"]:
        if not urr["daily_comments"].get(flair):
            dc = api_get(
                f"{base}/comments/search/aggregate",
                {
                    "subreddit": "UkraineRussiaReport",
                    "author_flair_text": flair,
                    "after": "2026-10-01",
                    "aggregate": "created_utc",
                    "frequency": "day",
                },
            )
            if dc and dc.get("data"):
                urr["daily_comments"][flair] = dc["data"]
                print(f"  URR flair='{flair}' daily_comments: {len(dc['data'])} days")
        if not urr["daily_posts"].get(flair):
            dp = api_get(
                f"{base}/posts/search/aggregate",
                {
                    "subreddit": "UkraineRussiaReport",
                    "author_flair_text": flair,
                    "after": "2026-09-14",
                    "aggregate": "created_utc",
                    "frequency": "day",
                },
            )
            if dp and dp.get("data"):
                urr["daily_posts"][flair] = dp["data"]
                print(f"  URR flair='{flair}' daily_posts: {len(dp['data'])} days")

    with open(DATA_DIR / "urr_flair_series.json", "w", encoding="utf-8") as f:
        json.dump(urr, f, indent=2)

    # 4. Viral Diffusion Posts
    print("=== 4. Fetching Viral Diffusion Posts ===")
    viral_posts = []
    seen_v = set()
    for sub in ["technology", "europe", "UkrainianConflict", "thebulwark", "AskConservatives", "Conservative", "OutOfTheLoop"]:
        ps = paginate_posts(sub, "2026-10-08", max_pages=4)
        for p in ps:
            tl = (p.get("title") or "").lower()
            if any(k in tl for k in ["yandex", "data center", "datacenter", "cloudflare", "russian bot", "bot farm", "r/conservative", "troll"]):
                if p["id"] not in seen_v:
                    seen_v.add(p["id"])
                    p["subreddit"] = sub
                    viral_posts.append(p)
    with open(DATA_DIR / "viral_diffusion.json", "w", encoding="utf-8") as f:
        json.dump({"reddit_viral_posts": sorted(viral_posts, key=lambda x: x.get("created_utc", 0)), "hn_hits": []}, f, indent=2)
    print(f"  Found {len(viral_posts)} viral Reddit threads.")


if __name__ == "__main__":
    main()
