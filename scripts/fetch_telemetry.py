#!/usr/bin/env python3
"""
Ultra-fast, non-blocking completion script for remaining Reddit datasets:
- arctic_shift_aggregates.json
- granular_posts.json
- viral_diffusion.json
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "PoliticalBotsResearch/1.0"}

SUBREDDIT_COHORTS = {
    "Conservative": "Primary Subject",
    "conspiracy": "Ideological / Geopolitical Treatment",
    "WayOfTheBern": "Ideological / Geopolitical Treatment",
    "EndlessWar": "Ideological / Geopolitical Treatment",
    "WalkAway": "Ideological / Geopolitical Treatment",
    "timpool": "Ideological / Geopolitical Treatment",
    "UkraineRussiaReport": "Conflict / Flair Natural Experiment",
    "UkrainianConflict": "Conflict / Flair Natural Experiment",
    "AskARussian": "RuNet / Russian Infrastructure",
    "pikabu": "RuNet / Russian Infrastructure",
    "tjournal_refugees": "RuNet / Russian Infrastructure",
    "politics": "Mainstream Political Control",
    "worldnews": "Mainstream Political Control",
    "AskConservatives": "Mainstream Political Control",
    "moderatepolitics": "Mainstream Political Control",
    "nfl": "Apolitical Baseline Control",
}


def fast_get(url: str, params: dict, timeout: float = 7.0):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_subreddit_posts_fast(subreddit: str, start_iso: str, max_pages: int = 20) -> list:
    """Fast indexed pagination of /api/posts/search (~0.3s per page)."""
    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    all_posts = []
    seen_ids = set()
    current_after = start_iso
    fields = (
        "id,created_utc,author,author_flair_text,title,domain,score,upvote_ratio,"
        "num_comments,removed_by_category,is_self,link_flair_text,selftext"
    )

    for _ in range(max_pages):
        res = fast_get(
            base_url,
            {
                "subreddit": subreddit,
                "after": current_after,
                "limit": "auto",
                "sort": "asc",
                "fields": fields,
            },
            timeout=8.0,
        )
        if not res or "data" not in res or not res["data"]:
            break

        batch = res["data"]
        new_items = 0
        max_ts = None
        for item in batch:
            pid = item.get("id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                st = item.get("selftext") or ""
                item["selftext_status"] = (
                    "removed"
                    if st == "[removed]"
                    else ("deleted" if st == "[deleted]" else ("empty" if st == "" else "present"))
                )
                item.pop("selftext", None)
                all_posts.append(item)
                new_items += 1
            ts = item.get("created_utc")
            if ts is not None and (max_ts is None or ts > max_ts):
                max_ts = ts

        if new_items == 0 or len(batch) < 25 or max_ts is None:
            break
        current_after = str(int(max_ts))
        time.sleep(0.15)

    return all_posts


def main():
    t0 = time.time()
    base = "https://arctic-shift.photon-reddit.com/api"

    # 1. Fetch daily post aggregates (fast) + Oct 1-12 comment & hourly aggregates
    agg_file = DATA_DIR / "arctic_shift_aggregates.json"
    print("=== 1. Fetching Subreddit Daily & Hourly Aggregations (Fast Single-Pass) ===")
    results = {
        "cohorts": SUBREDDIT_COHORTS,
        "daily_posts": {},
        "daily_comments": {},
        "hourly_posts": {},
        "hourly_comments": {},
    }

    for sub in SUBREDDIT_COHORTS:
        dp = fast_get(
            f"{base}/posts/search/aggregate",
            {"subreddit": sub, "after": "2026-09-10", "aggregate": "created_utc", "frequency": "day"},
            timeout=7.0,
        )
        if not dp or not dp.get("data"):
            dp = fast_get(
                f"{base}/posts/search/aggregate",
                {"subreddit": sub, "after": "2026-09-24", "aggregate": "created_utc", "frequency": "day"},
                timeout=7.0,
            )
        results["daily_posts"][sub] = dp.get("data", []) if dp else []

        dc = fast_get(
            f"{base}/comments/search/aggregate",
            {"subreddit": sub, "after": "2026-09-28", "aggregate": "created_utc", "frequency": "day"},
            timeout=6.5,
        )
        if not dc or not dc.get("data"):
            dc = fast_get(
                f"{base}/comments/search/aggregate",
                {"subreddit": sub, "after": "2026-10-01", "aggregate": "created_utc", "frequency": "day"},
                timeout=6.5,
            )
        results["daily_comments"][sub] = dc.get("data", []) if dc else []
        print(
            f"  r/{sub}: {len(results['daily_posts'][sub])} daily post days, "
            f"{len(results['daily_comments'][sub])} daily comment days"
        )

    hourly_targets = [
        "Conservative",
        "conspiracy",
        "WayOfTheBern",
        "EndlessWar",
        "WalkAway",
        "UkraineRussiaReport",
        "UkrainianConflict",
        "AskARussian",
        "pikabu",
        "tjournal_refugees",
        "AskConservatives",
        "moderatepolitics",
    ]
    for sub in hourly_targets:
        hp = fast_get(
            f"{base}/posts/search/aggregate",
            {"subreddit": sub, "after": "2026-10-01", "aggregate": "created_utc", "frequency": "hour"},
            timeout=6.5,
        )
        results["hourly_posts"][sub] = hp.get("data", []) if hp else []

        hc = fast_get(
            f"{base}/comments/search/aggregate",
            {"subreddit": sub, "after": "2026-10-04", "aggregate": "created_utc", "frequency": "hour"},
            timeout=6.5,
        )
        results["hourly_comments"][sub] = hc.get("data", []) if hc else []
        print(
            f"  [Hourly] r/{sub}: {len(results['hourly_posts'][sub])} post hrs, "
            f"{len(results['hourly_comments'][sub])} comment hrs"
        )

    with open(agg_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {agg_file} in {time.time() - t0:.1f}s")

    # 2. Ensure urr_flair_series.json has clean Oct 1-12 daily & hourly series for Pro Russia / Pro Ukraine / Neutral
    urr_file = DATA_DIR / "urr_flair_series.json"
    print("=== 2. Verifying r/UkraineRussiaReport Flair Series ===")
    urr_data = {"daily_posts": {}, "daily_comments": {}, "hourly_posts": {}, "hourly_comments": {}}
    for flair in ["Pro Russia", "Pro Ukraine", "Neutral"]:
        dp = fast_get(
            f"{base}/posts/search/aggregate",
            {
                "subreddit": "UkraineRussiaReport",
                "author_flair_text": flair,
                "after": "2026-09-14",
                "aggregate": "created_utc",
                "frequency": "day",
            },
            timeout=7.0,
        )
        dc = fast_get(
            f"{base}/comments/search/aggregate",
            {
                "subreddit": "UkraineRussiaReport",
                "author_flair_text": flair,
                "after": "2026-09-28",
                "aggregate": "created_utc",
                "frequency": "day",
            },
            timeout=7.0,
        )
        hp = fast_get(
            f"{base}/posts/search/aggregate",
            {
                "subreddit": "UkraineRussiaReport",
                "author_flair_text": flair,
                "after": "2026-10-01",
                "aggregate": "created_utc",
                "frequency": "hour",
            },
            timeout=7.0,
        )
        hc = fast_get(
            f"{base}/comments/search/aggregate",
            {
                "subreddit": "UkraineRussiaReport",
                "author_flair_text": flair,
                "after": "2026-10-04",
                "aggregate": "created_utc",
                "frequency": "hour",
            },
            timeout=7.0,
        )
        urr_data["daily_posts"][flair] = dp.get("data", []) if dp else []
        urr_data["daily_comments"][flair] = dc.get("data", []) if dc else []
        urr_data["hourly_posts"][flair] = hp.get("data", []) if hp else []
        urr_data["hourly_comments"][flair] = hc.get("data", []) if hc else []
        print(
            f"  Flair '{flair}': {len(urr_data['daily_posts'][flair])} post days, "
            f"{len(urr_data['daily_comments'][flair])} comment days"
        )
    with open(urr_file, "w", encoding="utf-8") as f:
        json.dump(urr_data, f, indent=2)

    # 3. Granular posts (fast indexed pagination)
    gran_file = DATA_DIR / "granular_posts.json"
    print("=== 3. Fetching Granular Post Records ===")
    gran_targets = {
        "Conservative": ("2026-09-10", 25),
        "WayOfTheBern": ("2026-09-10", 15),
        "EndlessWar": ("2026-09-10", 15),
        "WalkAway": ("2026-09-10", 15),
        "AskARussian": ("2026-09-10", 15),
        "AskConservatives": ("2026-09-10", 15),
        "moderatepolitics": ("2026-09-10", 15),
        "UkraineRussiaReport": ("2026-09-28", 20),
        "conspiracy": ("2026-09-28", 20),
        "pikabu": ("2026-09-28", 15),
    }
    store = {}
    for sub, (start_dt, max_p) in gran_targets.items():
        posts = fetch_subreddit_posts_fast(sub, start_dt, max_pages=max_p)
        store[sub] = posts
        print(f"  r/{sub}: {len(posts)} granular posts fetched")
    with open(gran_file, "w", encoding="utf-8") as f:
        json.dump(store, f)
    print(f"Saved {gran_file} in {time.time() - t0:.1f}s")

    # 4. Viral claim diffusion
    viral_file = DATA_DIR / "viral_diffusion.json"
    print("=== 4. Fetching Viral Claim Diffusion ===")
    viral_subs = [
        "technology",
        "europe",
        "UkrainianConflict",
        "thebulwark",
        "AskConservatives",
        "Conservative",
        "OutOfTheLoop",
        "SubredditDrama",
    ]
    viral_posts = []
    seen = set()
    for sub in viral_subs:
        posts = fetch_subreddit_posts_fast(sub, "2026-10-08", max_pages=6)
        for p in posts:
            tl = (p.get("title") or "").lower()
            if any(
                k in tl
                for k in [
                    "yandex",
                    "data center",
                    "datacenter",
                    "cloudflare",
                    "russian bot",
                    "bot farm",
                    "r/conservative",
                    "troll farm",
                ]
            ):
                if p["id"] not in seen:
                    seen.add(p["id"])
                    p["subreddit"] = sub
                    viral_posts.append(p)

    hn_hits = []
    for query in ["Yandex", "Russian data center", "AS13238"]:
        res = fast_get(
            "https://hn.algolia.com/api/v1/search_by_date",
            {"query": query, "numericFilters": "created_at_i>1791331200", "hitsPerPage": 100},
            timeout=6.0,
        )
        if res and "hits" in res:
            hn_hits.extend(res["hits"])

    with open(viral_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reddit_viral_posts": sorted(viral_posts, key=lambda x: x.get("created_utc", 0)),
                "hn_hits": hn_hits,
            },
            f,
            indent=2,
        )
    print(f"Saved {viral_file} ({len(viral_posts)} viral Reddit threads, {len(hn_hits)} HN hits)")
    print(f"=== DONE in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
