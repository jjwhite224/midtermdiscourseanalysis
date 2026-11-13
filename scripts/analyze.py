#!/usr/bin/env python3
"""Aggregate locally collected YouTube comments for the public website.

Input: data/private/comments.csv (never commit).
Output: data/summary.json (no comment text, IDs, or usernames).
Uses TF-IDF and scikit-learn K-Means; clusters are descriptive, not ideology labels.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STOP = set(ENGLISH_STOP_WORDS) | {
    "https", "http", "www", "youtube", "video", "channel", "just", "like",
    "really", "dont", "doesnt", "im", "youre", "thats", "yeah", "lol", "lmao",
    "com", "watch", "got", "gonna", "wanna", "thing", "things", "did", "didnt"
}
PHASE_ORDER = ("before", "election", "after")


def clean(text):
    text = unquote(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"&(?:amp|lt|gt|quot);", " ", text)
    return " ".join(text.lower().split())


def terms(texts, size=12):
    counter = Counter()
    for line in texts:
        words = set(re.findall(r"[a-z][a-z']{2,}", clean(line)))
        counter.update(w for w in words if w not in STOP and len(w) >= 3)
    return [{"term": term, "count": n} for term, n in counter.most_common(size)]


def main():
    input_path = DATA / "private" / "comments.csv"
    if not input_path.exists():
        raise SystemExit("No local comments.csv. Run: python scripts/collect.py --target 150")
    config = json.loads((DATA / "videos.json").read_text(encoding="utf-8"))
    report_path = DATA / "collection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    with input_path.open(newline="", encoding="utf-8") as f:
        source = list(csv.DictReader(f))
    # Dedupe by the source's stable comment ID (a comment may appear in one batch twice).
    by_id = {r["comment_id"]: r for r in source if (r.get("text") or "").strip()}
    rows = list(by_id.values())
    if not rows:
        raise SystemExit("No eligible comments; keeping previous public summary unchanged.")

    channel_phase = defaultdict(list)
    by_phase = defaultdict(list)
    for row in rows:
        if row["phase"] not in config["phases"] or row["channel"] not in config["channels"]:
            continue
        channel_phase[(row["channel"], row["phase"])].append(row)
        by_phase[row["phase"]].append(row)
    rows = [r for phase in PHASE_ORDER for r in by_phase[phase]]
    texts = [clean(r["text"]) for r in rows]

    # Baseline: word use in the sample. Presence per comment rather than raw repetition.
    terms_by_phase = [
        {"phase": phase, "terms": terms([r["text"] for r in by_phase[phase]])}
        for phase in PHASE_ORDER
    ]
    by_group = []
    for channel in config["channels"]:
        for phase in PHASE_ORDER:
            subset = channel_phase[(channel, phase)]
            by_group.append({
                "channel": channel, "phase": phase, "count": len(subset),
                "top_terms": terms([r["text"] for r in subset], 10)
            })

    # One shared model across every channel/phase, essential for fair comparison.
    cluster_info, allocations = [], []
    valid = [(r, text) for r, text in zip(rows, texts) if len(text.split()) >= 3]
    if len(valid) >= 12:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english", min_df=2 if len(valid) >= 25 else 1,
                max_df=0.9, ngram_range=(1, 2), sublinear_tf=True, max_features=3000
            )
            matrix = vectorizer.fit_transform([t for _, t in valid])
            k = min(6, max(2, len(valid) // 35))
            if matrix.shape[1] >= 2 and matrix.shape[0] >= k:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(matrix)
                vocabulary = vectorizer.get_feature_names_out()
                for idx, center in enumerate(model.cluster_centers_):
                    top = center.argsort()[::-1][:6]
                    cluster_info.append({
                        "id": idx, "label": "Cluster " + str(idx + 1),
                        "terms": [str(vocabulary[j]) for j in top]
                    })
                counts = Counter((row["channel"], row["phase"], int(label))
                                 for (row, _), label in zip(valid, labels))
                for channel in config["channels"]:
                    for phase in PHASE_ORDER:
                        for cluster in cluster_info:
                            allocations.append({
                                "channel": channel, "phase": phase,
                                "cluster_id": cluster["id"],
                                "count": counts[(channel, phase, cluster["id"])]
                            })
        except ValueError as exc:
            print("TF-IDF/K-Means skipped:", exc)

    outcome = {
        "status": "analyzed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_comments": len(rows),
        "phase_order": list(PHASE_ORDER),
        "phases": config["phases"],
        "channels": list(config["channels"]),
        "by_channel_phase": by_group,
        "terms_by_phase": terms_by_phase,
        "clusters": cluster_info,
        "cluster_allocations": allocations,
        "notes": (
            "Exploratory analysis of public top-level comments, selected via a stable hash "
            "within timestamps. Term counts indicate comments containing each word, not sentiment. "
            "K-Means clusters are unsupervised lexical groups, not ideological labels."
        ),
        "collection_report": [
            {key: entry.get(key) for key in
             ("channel", "phase", "video_id", "status", "selected", "eligible_scanned",
              "scan_complete", "stop_reason", "under_target")}
            for entry in report.get("reports", [])
        ]
    }
    (DATA / "summary.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(f"Analyzed {len(rows)} comments; {len(cluster_info)} exploratory clusters.")
    print("Only aggregate findings are in data/summary.json; no comment text was published.")


if __name__ == "__main__":
    main()
