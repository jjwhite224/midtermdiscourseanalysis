#!/usr/bin/env python3
"""Collect time-bounded, top-level YouTube comments for the 2026 midterm pilot.

Requires YOUTUBE_API_KEY. Writes raw comment text only to data/private/.
Uses publishedAt (not updatedAt), Eastern-time phase windows, API pagination,
and stable hash sampling. Does not collect names, handles or user IDs.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://www.googleapis.com/youtube/v3/"
FIELDS = ["comment_id", "video_id", "channel", "phase", "published_at_utc", "text", "likes"]


def utc_time(raw):
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def api_get(endpoint, params, key, attempts=3):
    url = API_BASE + endpoint + "?" + urlencode({**params, "key": key})
    for attempt in range(attempts):
        try:
            req = Request(url, headers={"User-Agent": "AcademicMidtermDiscoursePilot/0.1"})
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")[:300]
            if err.code not in (429, 500, 502, 503) or attempt == attempts - 1:
                raise RuntimeError(f"YouTube API {endpoint}: HTTP {err.code}: {detail}") from err
        except URLError as err:
            if attempt == attempts - 1:
                raise RuntimeError(f"YouTube API {endpoint}: {err}") from err
        time.sleep(2 ** attempt)


def verify_video(video_id, channel_id, start, end, key):
    payload = api_get("videos", {"part": "snippet", "id": video_id}, key)
    if not payload.get("items"):
        raise ValueError(f"Video unavailable: {video_id}")
    item = payload["items"][0]
    snip = item["snippet"]
    published = utc_time(snip["publishedAt"])
    if snip["channelId"] != channel_id:
        raise ValueError(f"Video {video_id} belongs to {snip['channelId']}, not {channel_id}")
    if not (start <= published < end):
        raise ValueError(f"Video {video_id} uploaded outside its assigned phase: {snip['publishedAt']}")
    return {"id": video_id, "title": snip["title"], "published_at_utc": snip["publishedAt"],
            "channel_id": snip["channelId"], "url": "https://www.youtube.com/watch?v=" + video_id}


def discover_video(spec, channel_id, start, end, key):
    """Find a candidate on the specified official channel, never a mirror upload."""
    query = spec.get("search_query") or "2026 midterm election"
    payload = api_get("search", {
        "part": "snippet", "channelId": channel_id, "q": query,
        "publishedAfter": start.isoformat().replace("+00:00", "Z"),
        "publishedBefore": end.isoformat().replace("+00:00", "Z"),
        "type": "video", "order": "relevance", "maxResults": 50
    }, key)
    choices = []
    for item in payload.get("items", []):
        vid = item.get("id", {}).get("videoId")
        if not vid:
            continue
        try:
            meta = verify_video(vid, channel_id, start, end, key)
        except ValueError:
            continue
        title = meta["title"].lower()
        score = sum(1 for token in ("midterm", "election", "poll", "senate", "house", "vot") if token in title)
        choices.append((score, meta))
    if not choices:
        raise ValueError(f"No suitable verified video found for {spec['channel']} / {spec['phase']}")
    choices.sort(key=lambda x: (-x[0], x[1]["published_at_utc"], x[1]["id"]))
    selected = choices[0][1]
    selected["auto_selected"] = True
    selected["review_required"] = True
    selected["other_candidates"] = [
        {"id": m["id"], "title": m["title"], "score": score} for score, m in choices[1:6]
    ]
    return selected


def collect_video(video, start, end, key, target, max_pages):
    found = []
    seen = set()
    reached_boundary = False
    pages = 0
    token = None
    while pages < max_pages:
        params = {"part": "snippet", "videoId": video["id"], "maxResults": 100,
                  "order": "time", "textFormat": "plainText"}
        if token:
            params["pageToken"] = token
        response = api_get("commentThreads", params, key)
        pages += 1
        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            data = top["snippet"]
            published = utc_time(data["publishedAt"])
            if published < start:
                reached_boundary = True
                continue
            if published >= end:
                continue
            comment_id = top["id"]
            if comment_id in seen:
                continue
            seen.add(comment_id)
            body = (data.get("textDisplay") or "").strip()
            if not body:
                continue
            found.append({
                "comment_id": comment_id, "video_id": video["id"],
                "channel": video["channel"], "phase": video["phase"],
                "published_at_utc": data["publishedAt"], "text": body,
                "likes": data.get("likeCount", 0)
            })
        token = response.get("nextPageToken")
        if reached_boundary or not token:
            break
    # Stable-hash sample avoids cherry-picking highest-liked/recent comments.
    selected = sorted(found, key=lambda x: hashlib.sha256(
        (x["video_id"] + ":" + x["comment_id"]).encode("utf-8")).digest())[:target]
    return selected, {
        "eligible_scanned": len(found), "selected": len(selected),
        "pages_scanned": pages,
        "scan_complete": reached_boundary or not token,
        "stop_reason": "phase_boundary" if reached_boundary else ("exhausted" if not token else "page_cap"),
        "under_target": len(selected) < target
    }


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=150, help="Desired 100-200 comments per video")
    parser.add_argument("--max-pages", type=int, default=150, help="Maximum pages per video; may reduce coverage")
    parser.add_argument("--dry-run", action="store_true", help="Verify video metadata without collecting comments")
    args = parser.parse_args()
    if not (1 <= args.target <= 200):
        parser.error("--target must be 1-200")
    if args.max_pages < 1:
        parser.error("--max-pages must be >=1")
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        sys.exit("Set YOUTUBE_API_KEY in your environment (do not commit it).")

    config = json.loads((ROOT / "data" / "videos.json").read_text(encoding="utf-8"))
    zone = ZoneInfo(config["timezone"])
    resolved, reports, all_rows = [], [], []
    for spec in config["videos"]:
        bounds = config["phases"][spec["phase"]]
        start = datetime.fromisoformat(bounds["start"]).replace(tzinfo=zone).astimezone(timezone.utc)
        end = datetime.fromisoformat(bounds["end"]).replace(tzinfo=zone).astimezone(timezone.utc)
        channel_id = config["channels"][spec["channel"]]["channel_id"]
        print(f"Checking {spec['channel']} / {spec['phase']}...", flush=True)
        try:
            if spec.get("id"):
                video = verify_video(spec["id"], channel_id, start, end, key)
                video["auto_selected"] = False
            else:
                video = discover_video(spec, channel_id, start, end, key)
            video.update(channel=spec["channel"], phase=spec["phase"], source=spec.get("source"))
            resolved.append(video)
            if args.dry_run:
                reports.append({"channel": spec["channel"], "phase": spec["phase"],
                                "video_id": video["id"], "status": "verified_only"})
                continue
            rows, metrics = collect_video(video, start, end, key, args.target, args.max_pages)
            all_rows.extend(rows)
            reports.append({"channel": spec["channel"], "phase": spec["phase"],
                            "video_id": video["id"], "status": "collected", **metrics})
            print(f"  {len(rows)} collected; {metrics['stop_reason']}", flush=True)
        except (RuntimeError, ValueError) as error:
            print(f"  WARNING: {error}", file=sys.stderr)
            reports.append({"channel": spec["channel"], "phase": spec["phase"],
                            "video_id": spec.get("id"), "status": "error", "error": str(error)})

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / "resolved_videos.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    (data / "collection_report.json").write_text(
        json.dumps({"collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "target_per_video": args.target, "dry_run": args.dry_run,
                    "reports": reports}, indent=2) + "\n", encoding="utf-8")
    if not args.dry_run:
        private = data / "private"
        private.mkdir(exist_ok=True)
        with (private / "comments.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} local comments; {len(resolved)} verified videos.")
    print("Review auto-selected video titles before using the findings.")


if __name__ == "__main__":
    run()
