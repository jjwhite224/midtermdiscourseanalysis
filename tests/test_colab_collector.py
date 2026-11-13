"""Offline regression tests for the Colab's embedded YouTube comment collector.

No API keys or network access needed. We extract only function definitions
from the notebook and substitute a fake YouTube API response.
"""
import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kmeans_midterm_walkthrough.ipynb"


def collector_namespace():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell.get("source", [])) for cell in cells if cell["cell_type"] == "code"]
    helpers = next(source for source in sources if "def verify_selection(" in source)
    collecting = next(source for source in sources if "def collect_one_video(" in source)
    ns = {}
    exec(compile(helpers, "notebook_api_helpers", "exec"), ns)
    exec(compile(collecting, "notebook_collector", "exec"), ns)
    ns.update({
        "TIMEZONE": "America/New_York",
        "PHASE_WINDOWS": {
            "election": ("2026-11-03T00:00:00", "2026-11-04T00:00:00")
        },
        "CHANNEL_IDS": {"Fox News": "official-fox-id"},
        "REQUIRE_VIDEO_UPLOAD_IN_PHASE": True,
        "MAX_PAGES_PER_VIDEO": 3,
        "TARGET_PER_VIDEO": 2,
        "MIN_LIKES": 0,
        "MIN_WORDS_AT_COLLECTION": 1,
        "SAMPLING_METHOD": "hash",
    })
    return ns


def thread(cid, stamp, text="Election voters and results"):
    return {
        "snippet": {
            "topLevelComment": {
                "id": cid,
                "snippet": {
                    "publishedAt": stamp,
                    "textDisplay": text,
                    "likeCount": 0,
                },
            }
        }
    }


class ColabCollectorTest(unittest.TestCase):
    def test_video_verification_rejects_wrong_channel(self):
        ns = collector_namespace()
        ns["youtube_api"] = lambda endpoint, params, key: {
            "items": [{"snippet": {
                "channelId": "another-channel",
                "title": "Election night",
                "publishedAt": "2026-11-04T02:00:00Z",
            }}]
        }
        with self.assertRaisesRegex(ValueError, "not uploaded"):
            ns["verify_selection"]({
                "channel": "Fox News",
                "phase": "election",
                "video_id": "1FBFCOrpE-o"
            }, "not-a-real-key")

    def test_actual_comment_time_controls_phase(self):
        ns = collector_namespace()
        calls = []
        responses = [
            {
                "items": [
                    thread("after", "2026-11-04T06:00:00Z"),
                    thread("in-window1", "2026-11-04T02:00:00Z"),
                ],
                "nextPageToken": "next",
            },
            {
                "items": [
                    thread("in-window2", "2026-11-03T17:00:00Z"),
                    thread("before", "2026-11-02T18:00:00Z"),
                ],
            },
        ]

        def fake_api(endpoint, params, key):
            self.assertEqual(endpoint, "commentThreads")
            calls.append(params)
            return responses.pop(0)

        ns["youtube_api"] = fake_api
        video = {"channel": "Fox News", "phase": "election", "video_id": "1FBFCOrpE-o"}
        rows, report = ns["collect_one_video"](video, "fake")
        self.assertEqual({r["comment_id"] for r in rows},
                         {"in-window1", "in-window2"})
        self.assertEqual(report["eligible_scanned"], 2)
        self.assertEqual(report["pages_scanned"], 2)
        self.assertEqual(report["stop_reason"], "phase_boundary")
        self.assertTrue(report["scan_complete"])
        self.assertEqual(calls[1]["pageToken"], "next")

    def test_filters_and_sample_are_reproducible(self):
        ns = collector_namespace()
        ns["MIN_LIKES"] = 1
        responses = [{
            "items": [
                {
                    **thread("too-few-likes", "2026-11-03T15:00:00Z"),
                },
                {
                    "snippet": {
                        "topLevelComment": {
                            "id": "keep",
                            "snippet": {
                                "publishedAt": "2026-11-03T16:00:00Z",
                                "textDisplay": "A midterm election comment",
                                "likeCount": 2
                            },
                        }
                    }
                },
            ]
        }]
        ns["youtube_api"] = lambda endpoint, params, key: responses[0]
        video = {"channel": "Fox News", "phase": "election", "video_id": "1FBFCOrpE-o"}
        first, _ = ns["collect_one_video"](video, "fake")
        second, _ = ns["collect_one_video"](video, "fake")
        self.assertEqual(first, second)
        self.assertEqual([r["comment_id"] for r in first], ["keep"])


if __name__ == "__main__":
    unittest.main()
