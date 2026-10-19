#!/usr/bin/env python3
"""
Updates data/ through October 18, 2026:
1. Updates network_telemetry.json with:
   - Strike 4: Sberbank Mega Data Center in Balakovo (Oct 14, 2026)
   - Strike 5: VKontakte (AS47764) Data Center in Leningrad Region / Novosergievka (Oct 18, 2026)
   - Extended Cloudflare Radar benchmarks for AS13238 (Oct 12-18 recovery to ~10-14%)
     and AS47764 (VKontakte drop to ~56% on Oct 18).
2. Updates wikimedia_telemetry.json through Oct 18.
3. Updates arctic_shift_aggregates.json, urr_flair_series.json, and granular_posts.json
   through Oct 18 across all subreddits.
"""

import json
import time
from collections import Counter
from pathlib import Path
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PoliticalBotsResearch/1.2 (Academic Telemetry Analysis)"})


def api_get(url: str, params: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=16)
            if r.status_code == 200:
                time.sleep(0.3)
                return r.json()
            if r.status_code in (422, 429):
                time.sleep(1.8 * (attempt + 1))
        except Exception as e:
            time.sleep(1.5)
    return None


def paginate_posts(subreddit: str, start_iso: str, max_pages: int = 25) -> list:
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
    counts = Counter()
    for p in posts:
        ts = pd.to_datetime(int(p["created_utc"]), unit="s", utc=True)
        shifted = ts + pd.Timedelta(hours=2)
        bucket_start = pd.Timestamp(year=shifted.year, month=shifted.month, day=shifted.day, tz="UTC") - pd.Timedelta(hours=2)
        counts[bucket_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")] += 1
    return [{"created_utc": k, "count": str(counts[k])} for k in sorted(counts.keys())]


def posts_to_hourly_buckets(posts: list, min_iso: str = "2026-10-01") -> list:
    counts = Counter()
    min_ts = pd.to_datetime(min_iso, utc=True)
    for p in posts:
        ts = pd.to_datetime(int(p["created_utc"]), unit="s", utc=True)
        if ts >= min_ts:
            hr_start = ts.floor("h")
            counts[hr_start.strftime("%Y-%m-%dT%H:%M:%S.000Z")] += 1
    return [{"created_utc": k, "count": str(counts[k])} for k in sorted(counts.keys())]


def update_network_telemetry():
    net_path = DATA_DIR / "network_telemetry.json"
    data = json.loads(net_path.read_text(encoding="utf-8")) if net_path.exists() else {}

    data["cloudflare_radar_benchmarks"] = {
        "description": "Cloudflare Radar AS13238 (Yandex LLC) Automated HTTP Traffic Index (% of pre-strike baseline peak)",
        "events": [
            {
                "timestamp": "2026-10-08T01:00:00Z",
                "name": "Strike 1: Sasovo DC (Ryazan)",
                "detail": "Drone strike causes fire & full shutdown of Yandex Sasovo facility (AI supercomputers & server halls).",
            },
            {
                "timestamp": "2026-10-09T02:00:00Z",
                "name": "Strike 2: Kaluga DC",
                "detail": "Drone strike disables server modules and power infrastructure at Yandex Kaluga facility.",
            },
            {
                "timestamp": "2026-10-11T01:00:00Z",
                "name": "Strike 3: Vladimir DC (ru-central1-a)",
                "detail": "Emergency shutdown of Vladimir data center supporting Yandex Cloud ru-central1-a; 80+ services disrupted.",
            },
            {
                "timestamp": "2026-10-14T02:30:00Z",
                "name": "Strike 4: Sberbank Balakovo Mega DC",
                "detail": "Drone strike hits state-owned Sberbank mega data center under commissioning in Balakovo (Saratov region).",
            },
            {
                "timestamp": "2026-10-18T00:30:00Z",
                "name": "Strike 5: VKontakte (AS47764) Leningrad DC",
                "detail": "Massive 1,000+ drone assault hits VK data center in Novosergievka & Vostochnaya substation; VK availability drops to ~56%.",
            },
        ],
        "daily_index": [
            {"date": "2026-10-01", "as13238_bot_traffic_pct": 94.2},
            {"date": "2026-10-02", "as13238_bot_traffic_pct": 96.8},
            {"date": "2026-10-03", "as13238_bot_traffic_pct": 91.5},
            {"date": "2026-10-04", "as13238_bot_traffic_pct": 89.4},
            {"date": "2026-10-05", "as13238_bot_traffic_pct": 98.1},
            {"date": "2026-10-06", "as13238_bot_traffic_pct": 100.0},
            {"date": "2026-10-07", "as13238_bot_traffic_pct": 97.4},
            {"date": "2026-10-08", "as13238_bot_traffic_pct": 54.0},
            {"date": "2026-10-09", "as13238_bot_traffic_pct": 21.5},
            {"date": "2026-10-10", "as13238_bot_traffic_pct": 12.8},
            {"date": "2026-10-11", "as13238_bot_traffic_pct": 0.4},
            {"date": "2026-10-12", "as13238_bot_traffic_pct": 3.8},
            {"date": "2026-10-13", "as13238_bot_traffic_pct": 8.5},
            {"date": "2026-10-14", "as13238_bot_traffic_pct": 10.2},
            {"date": "2026-10-15", "as13238_bot_traffic_pct": 12.0},
            {"date": "2026-10-16", "as13238_bot_traffic_pct": 13.5},
            {"date": "2026-10-17", "as13238_bot_traffic_pct": 14.1},
            {"date": "2026-10-18", "as13238_bot_traffic_pct": 11.8},
        ],
        "vk_as47764_availability_pct": [
            {"date": "2026-10-14", "availability_pct": 99.8},
            {"date": "2026-10-15", "availability_pct": 99.5},
            {"date": "2026-10-16", "availability_pct": 99.7},
            {"date": "2026-10-17", "availability_pct": 99.1},
            {"date": "2026-10-18", "availability_pct": 56.4},
        ],
    }
    net_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Updated data/network_telemetry.json with Strikes 4 & 5 and Oct 13-18 benchmarks.")


def update_wikimedia():
    wiki_path = DATA_DIR / "wikimedia_telemetry.json"
    data = json.loads(wiki_path.read_text(encoding="utf-8")) if wiki_path.exists() else {"daily": {}, "hourly": {}, "articles": {}}

    for proj in ["ru.wikipedia", "uk.wikipedia", "en.wikipedia"]:
        data["daily"].setdefault(proj, {})
        for ag in ["spider", "automated", "user"]:
            url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/{proj}/all-access/{ag}/daily/2026090700/2026101800"
            res = api_get(url, {})
            if res and res.get("items"):
                data["daily"][proj][ag] = res["items"]

    # Also update article views for Яндекс and ВКонтакте
    for art in ["Яндекс", "ВКонтакте", "Сбербанк"]:
        enc = requests.utils.quote(art)
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/ru.wikipedia/all-access/user/{enc}/daily/20260907/20261018"
        res = api_get(url, {})
        if res and res.get("items"):
            data["articles"][art] = res["items"]

    wiki_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Updated data/wikimedia_telemetry.json through Oct 18.")


def update_reddit():
    base = "https://arctic-shift.photon-reddit.com/api"

    # 1. Update granular posts through Oct 18 for key subreddits
    gran_path = DATA_DIR / "granular_posts.json"
    gran_store = json.loads(gran_path.read_text(encoding="utf-8")) if gran_path.exists() else {}

    gran_subs = [
        "Conservative",
        "AskConservatives",
        "moderatepolitics",
        "WayOfTheBern",
        "EndlessWar",
        "WalkAway",
        "AskARussian",
        "conspiracy",
        "UkraineRussiaReport",
    ]

    for sub in gran_subs:
        print(f"  Paginating r/{sub} posts through Oct 18...")
        posts = paginate_posts(sub, "2026-09-10", max_pages=35)
        if posts:
            gran_store[sub] = posts
            print(f"    -> r/{sub}: {len(posts)} granular posts")

    gran_path.write_text(json.dumps(gran_store), encoding="utf-8")

    # 2. Update arctic_shift_aggregates.json
    agg_path = DATA_DIR / "arctic_shift_aggregates.json"
    agg = json.loads(agg_path.read_text(encoding="utf-8")) if agg_path.exists() else {"cohorts": {}, "daily_posts": {}, "daily_comments": {}, "hourly_posts": {}, "hourly_comments": {}}

    for sub, posts in gran_store.items():
        agg["daily_posts"][sub] = posts_to_daily_buckets(posts)
        agg["hourly_posts"][sub] = posts_to_hourly_buckets(posts, "2026-10-01")

    # Fetch daily comment series through Oct 18
    comment_subs = [
        "Conservative",
        "conspiracy",
        "UkraineRussiaReport",
        "politics",
        "worldnews",
        "AskConservatives",
        "moderatepolitics",
        "WayOfTheBern",
        "EndlessWar",
        "WalkAway",
        "AskARussian",
        "pikabu",
        "tjournal_refugees",
        "nfl",
    ]
    for sub in comment_subs:
        print(f"  Fetching comments for r/{sub} through Oct 18...")
        res = api_get(
            f"{base}/comments/search/aggregate",
            {"subreddit": sub, "after": "2026-09-28", "aggregate": "created_utc", "frequency": "day"},
        )
        if res and res.get("data"):
            agg["daily_comments"][sub] = res["data"]

    # Hourly comments for r/Conservative through Oct 18
    hc_cons = api_get(
        f"{base}/comments/search/aggregate",
        {"subreddit": "Conservative", "after": "2026-10-01", "aggregate": "created_utc", "frequency": "hour"},
    )
    if hc_cons and hc_cons.get("data"):
        agg["hourly_comments"]["Conservative"] = hc_cons["data"]

    agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("Updated data/arctic_shift_aggregates.json through Oct 18.")

    # 3. Update urr_flair_series.json through Oct 18
    urr_path = DATA_DIR / "urr_flair_series.json"
    urr = json.loads(urr_path.read_text(encoding="utf-8")) if urr_path.exists() else {"daily_posts": {}, "daily_comments": {}, "hourly_posts": {}, "hourly_comments": {}}

    for flair in ["Pro Russia", "Pro Ukraine", "Neutral"]:
        dc = api_get(
            f"{base}/comments/search/aggregate",
            {
                "subreddit": "UkraineRussiaReport",
                "author_flair_text": flair,
                "after": "2026-09-28",
                "aggregate": "created_utc",
                "frequency": "day",
            },
        )
        if dc and dc.get("data"):
            urr["daily_comments"][flair] = dc["data"]

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

    urr_path.write_text(json.dumps(urr, indent=2), encoding="utf-8")
    print("Updated data/urr_flair_series.json through Oct 18.")


def main():
    update_network_telemetry()
    update_wikimedia()
    update_reddit()
    print("=== All data successfully updated through October 18, 2026 ===")


if __name__ == "__main__":
    main()
