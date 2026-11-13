# Before the Vote, After the Result

Pilot for DATA 78000 — AI and Machine Learning for Artists and Humanists (Fall 2026).

**Research question:** How did language and topics in YouTube comment communities change before, during, and after the November 3, 2026 US midterm elections?

This pilot compares **HasanAbi** and **Fox News** across three phases (one video per phase, six videos total), aiming for **100–200 top-level comments per video**. The selection file has five independently sourced YouTube video IDs; the Fox pre-election upload is **intentionally unfilled** and resolved from the official channel with the YouTube Data API. Never interpret a placeholder as a verified video.

## Quick start

1. Create a YouTube Data API v3 key using [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com).
2. Put the key in the **local environment**; never commit it:
   - macOS/Linux: `export YOUTUBE_API_KEY="your-key"`
   - Windows PowerShell: `$env:YOUTUBE_API_KEY="your-key"`
3. Run: `python scripts/collect.py --target 150`
4. Review `data/collection_report.json` and `data/resolved_videos.json` before analyzing.
5. Install ML dependencies: `python -m pip install -r requirements.txt`
6. Run: `python scripts/analyze.py`
7. Preview: `python -m http.server 8000`, then open http://localhost:8000
8. To publish, commit **only** `data/summary.json`, `data/collection_report.json`, and any reviewed video metadata. Enable GitHub Pages from the root of `main` in repository Settings → Pages. **Never commit** `data/private/`.

## Collect and edit a YouTube dataset directly in Google Colab

**[Open the complete notebook in Google Colab](https://colab.research.google.com/github/jjwhite224/midtermdiscourseanalysis/blob/main/notebooks/kmeans_midterm_walkthrough.ipynb)**. The notebook now includes an **editable Colab dataset builder before Step 2**, so you do not need to separately run scripts/collect.py.

1. In [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com), enable YouTube Data API v3 and create a key.
2. In Colab's **Secrets** panel (key icon), add a secret named `YOUTUBE_API_KEY` and allow notebook access. The notebook can also prompt you for the key with hidden input. Never save a key in a code cell.
3. In the **dataset settings** cell, edit `VIDEO_SELECTION` (video IDs and enabled flags), `CHANNEL_IDS` (official uploader IDs), `PHASE_WINDOWS`, `TARGET_PER_VIDEO`, `MAX_PAGES_PER_VIDEO`, `MIN_LIKES`, `MIN_WORDS_AT_COLLECTION`, and `SAMPLING_METHOD`.
4. Run the import, configuration, verification, and collection cells **in order**. Check the video-title preview before collecting; the pre-election Fox News candidate is a short and may not be the best comparison.
5. Inspect group counts and any incomplete scans. The collected DataFrame is already in memory as `df`, and the raw CSV plus configuration/collection metadata are saved under `data/private/` **inside the Colab runtime**. The notebook includes an optional, commented-out download command.
6. Continue from **Step 2** through stop-word filtering, the elbow method, clustering, and visualizations. You can change the collection settings and rerun the collector to build a different dataset.

The default configuration is six videos (HasanAbi vs Fox News), 150 comments per video, with the same Eastern-time before/election/after windows as the initial pilot. The collector checks uploader ownership and upload dates against YouTube's API, then filters **each comment's publication timestamp**. It records shortfalls and page-cap warnings and does not collect author names or handles. Collection runs **only when you run the notebook cells**, and it does **not** overwrite the public GitHub Pages results.

**Privacy:** Colab runtime files are temporary. The private CSV contains comment bodies and IDs, so do not commit it or executed notebook outputs with identifiable excerpts. You can store a private local copy and publish aggregate results separately.

## Classroom-style clustering walkthrough

For a narrated, **step-by-step K-Means exercise** based on Professor Mandiberg's [Iris clustering demo](https://github.com/mandiberg/DATA-78000/tree/main/data_clustering), use either:

- [Jupyter notebook — notebooks/kmeans_midterm_walkthrough.ipynb](notebooks/kmeans_midterm_walkthrough.ipynb) (markdown explanations + one numbered code cell per major step)
- [Plain Python version — scripts/clustering_walkthrough.py](scripts/clustering_walkthrough.py) (same operations with numbered comments)

The tutorial is independent of the production scripts/analyze.py: it will not change the published website. It teaches **load → clean → standard/custom stop words → inspect vocabulary before/after → TF-IDF numerical features → elbow method → choose k → fit K-Means → inspect centroids and private examples → plot normalized groups → visualize 2D projection → interpret**. In **Step 3A**, edit `CUSTOM_STOP_WORDS` and optionally enable `REMOVE_REVIEW_WORDS`; never remove political terms solely to make clusters appear cleaner.

Run from the project root:

    python -m pip install -r requirements-notebook.txt
    python scripts/collect.py --target 150
    python scripts/clustering_walkthrough.py

Alternatively, open the notebook in Jupyter/Colab and run its cells in order. **The earlier GitHub Actions run did not save the private comment text to GitHub.** The collector needs to run locally again, or you can upload your privately held CSV into a temporary Colab notebook session. Do not commit the private CSV or executed notebook outputs containing comment text.

## No-install cloud collection (GitHub Actions)

Once your API key exists, open the repository's **Settings → Secrets and variables → Actions → New repository secret**. Name the secret **`YOUTUBE_API_KEY`** and paste the API key. Then go to **Actions → Collect and publish aggregate comment analysis → Run workflow**. Enter a target of **150**. The workflow collects locally on a temporary GitHub runner, runs the ML analysis, and commits only summary JSON and video metadata. This does not automatically turn on Pages; enable **Settings → Pages → Deploy from a branch → main / (root)** separately.

The first run will fill in the missing Fox News pre-election video through official-channel API discovery. **Review the auto-selected title in `data/resolved_videos.json` before interpreting results.** If no suitable video is found, the collector flags an error instead of inventing an ID.

## What the collector does

- Uses YouTube Data API `videos.list`, `search.list` (only for unresolved videos), and `commentThreads.list`.
- Verifies the expected uploading channel, the video date in **America/New_York**, and each comment's **publishedAt** timestamp.
- Keeps only top-level comments published during their video's assigned phase, uses stable-hash sampling up to the target, and writes raw comment text locally to an ignored CSV.
- Paginates by newest-first until crossing the phase's start boundary, reaching the page cap, or exhausting comments; warns when coverage is incomplete.
- No sample comments or findings are fabricated. The public site clearly says no results are available until the pipeline runs.

## Phase boundaries (New York time)

- **Before:** October 27–November 2, 2026 (end exclusive: Nov 3, 00:00)
- **Election Day:** November 3, 2026 (end exclusive: Nov 4, 00:00)
- **After:** November 4–10, 2026 (end exclusive: Nov 11, 00:00)

## Research limitations

The API cannot retrospectively recover removed comments or guarantee a representative audience sample; uploads and moderation differ by channel. This is an **observational, descriptive comparison**, not a measurement of voter beliefs or algorithmic causation. Commenters may not live in the US or be voters. One video per channel per phase makes conclusions preliminary. Keep this project privacy-conscious: usernames and profile IDs are not collected, and no individual comment bodies are published. Check [YouTube API Services Terms](https://developers.google.com/youtube/terms/api-services-terms-of-service) and [Developer Policies](https://developers.google.com/youtube/terms/developer-policies) before sharing any derived dataset.

## Video sourcing

- HasanAbi uploads: indexed in [Arcmira's HasanAbi channel archive](https://arcmira.com/yt/@HasanAbi).
- Fox News Election Day and post-election uploads: indexed in the public [viralshooter video metadata archive](https://viralshooter.com/).
- The collector verifies IDs, original channel IDs, and publication dates against the YouTube API before including them.

## Directory structure

- `data/videos.json` — planned, sourced video selections
- `data/resolved_videos.json` — collector-verified video metadata (after running)
- `data/private/comments.csv` — **local only**, never publish
- `data/collection_report.json` — collection completeness diagnostics
- `data/summary.json` — aggregate public chart data, generated by analysis
- `scripts/collect.py` — timestamp-aware YouTube comment collector
- `scripts/analyze.py` — TF-IDF and K-Means analysis with aggregate output
- `index.html` — interactive GitHub Pages story
