"""Offline tests for time-window correctness and YouTube API parsing."""
import importlib.util
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "collect.py"
spec = importlib.util.spec_from_file_location("collector", SOURCE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def top(cid, when, message):
    return {"snippet": {"topLevelComment": {"id": cid, "snippet": {
        "publishedAt": when, "textDisplay": message, "likeCount": 1
    }}}}


class CollectorTest(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc)
        self.end = datetime(2026, 11, 4, 5, 0, tzinfo=timezone.utc)

    def test_video_channel_and_period_validation(self):
        valid = {"items": [{"snippet": {"channelId": "channel-1",
            "title": "Election Night", "publishedAt": "2026-11-04T00:30:00Z"}}]}
        with patch.object(collector, "api_get", return_value=valid):
            result = collector.verify_video("1FBFCOrpE-o", "channel-1", self.start, self.end, "fake")
            self.assertEqual(result["title"], "Election Night")
            with self.assertRaises(ValueError):
                collector.verify_video("1FBFCOrpE-o", "not-that-channel", self.start, self.end, "fake")

    def test_timestamp_filter_uses_publication_date_not_video_date(self):
        responses = [
            {"items": [
                top("late", "2026-11-04T06:00:00Z", "After election window"),
                top("a", "2026-11-04T01:00:00Z", "Inside the election day"),
            ], "nextPageToken": "page2"},
            {"items": [
                top("b", "2026-11-03T14:00:00Z", "Voters today are voting"),
                top("early", "2026-11-02T19:00:00Z", "Before the election day")
            ]}
        ]
        vid = {"id": "abc12345678", "channel": "Fox News", "phase": "election"}
        with patch.object(collector, "api_get", side_effect=responses):
            rows, report = collector.collect_video(vid, self.start, self.end, "fake", 150, 10)
        self.assertEqual({row["comment_id"] for row in rows}, {"a", "b"})
        self.assertEqual(report["pages_scanned"], 2)
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["stop_reason"], "phase_boundary")

    def test_sample_is_deterministic(self):
        comments = [top(str(i), "2026-11-03T15:00:00Z", "This is a valid comment") for i in range(12)]
        response = {"items": comments}
        vid = {"id": "abc12345678", "channel": "HasanAbi", "phase": "election"}
        with patch.object(collector, "api_get", return_value=response):
            first, _ = collector.collect_video(vid, self.start, self.end, "fake", 4, 2)
            second, _ = collector.collect_video(vid, self.start, self.end, "fake", 4, 2)
        self.assertEqual([r["comment_id"] for r in first],
                         [r["comment_id"] for r in second])
        self.assertEqual(len(first), 4)


if __name__ == "__main__":
    unittest.main()
