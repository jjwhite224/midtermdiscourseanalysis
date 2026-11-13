#!/usr/bin/env python3
"""Classroom-style K-Means tutorial for midterm YouTube comments.
Run from the project root with private data/private/comments.csv present.
This script does not change data/summary.json or the website.
"""

# Step 1 — Import the Python libraries

import html
import re
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import silhouette_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print("Imports complete.")


# Step 2 — Load and inspect the dataset

paths = [Path("data/private/comments.csv"),
         Path("../data/private/comments.csv"),
         Path("comments.csv")]
paths = [p for p in paths if p.exists()]
if not paths:
    raise FileNotFoundError(
        "Collect comments locally with scripts/collect.py, "
        "or upload private comments.csv to this notebook session."
    )
df = pd.read_csv(paths[0])
required = {"comment_id", "channel", "phase", "text"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing CSV columns: {sorted(missing)}")
df = df.drop_duplicates(subset="comment_id").dropna(subset=["text"]).copy()
print("Number of unique comments:", len(df))
print(pd.crosstab(df["channel"], df["phase"]).to_string())


# Step 3 — Clean the text

def clean_text(value):
    value = html.unescape(str(value)).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

df["text_clean"] = df["text"].apply(clean_text)
df["word_count"] = df["text_clean"].str.split().str.len()
eligible = df.loc[df["word_count"] >= 3].copy().reset_index(drop=True)
print("Eligible comments:", len(eligible))
print("Excluded short comments:", len(df) - len(eligible))
print(pd.crosstab(eligible["channel"], eligible["phase"]).to_string())
if len(eligible) < 12:
    raise ValueError("At least 12 eligible comments are needed.")


# Step 3A — Standard stop words + your own exclusion list

# Mandiberg's custom stop-word approach, adapted to scikit-learn:
# 1. Start with a standard stop-word set.
STANDARD_STOP_WORDS = set(ENGLISH_STOP_WORDS)

# 2. Add your own words. Change this list based on what you observe.
CUSTOM_STOP_WORDS = {
    "hasan", "hasanabi", "fox", "youtube", "channel", "video",
    "subscribe", "subscribed", "thumbnail",
    "lol", "lmao", "yeah", "gonna", "wanna", "im", "ive",
    "dont", "doesnt", "didnt", "youre", "thats", "grokspx"
}

# 3. These might be topical references, jokes, or actual research findings.
# Don't remove them unless a private review justifies it.
REMOVE_REVIEW_WORDS = False
REVIEW_ONLY_WORDS = {"rachel", "outfit", "collab", "bluke", "blexas"}

# 4. Union the sets, as in the professor's custom_stop_words_pattern.py.
STOP_WORDS = STANDARD_STOP_WORDS.union(CUSTOM_STOP_WORDS)
if REMOVE_REVIEW_WORDS:
    STOP_WORDS = STOP_WORDS.union(REVIEW_ONLY_WORDS)

# 5. Show what will change. A token is a 2+-character word-like piece.
token_pattern = re.compile(r"(?u)\b\w\w+\b")
def words_in(text):
    return token_pattern.findall(text)

before_counts = Counter(
    word for comment in eligible["text_clean"] for word in words_in(comment)
)
after_counts = Counter({
    word: count for word, count in before_counts.items() if word not in STOP_WORDS
})
removed = sum(before_counts.values()) - sum(after_counts.values())

print("English stop words:", len(STANDARD_STOP_WORDS))
print("Custom words:", sorted(CUSTOM_STOP_WORDS))
print("Removed token occurrences:", removed,
      f"of {sum(before_counts.values())} ({removed / max(1, sum(before_counts.values())):.1%})")
print("\nMost frequent words BEFORE filtering:")
print(before_counts.most_common(15))
print("\nMost frequent words AFTER filtering:")
print(after_counts.most_common(15))

# Teaching example: show a made-up sentence, not someone's private comment.
example = "Hasan just talked about the Senate election; I voted today!"
example_tokens = words_in(clean_text(example))
print("\nSynthetic example before:", example_tokens)
print("Synthetic example after:",
      [token for token in example_tokens if token not in STOP_WORDS])


# Step 4 — Turn remaining words into numerical features (TF-IDF)

vectorizer = TfidfVectorizer(
    stop_words=sorted(STOP_WORDS),
    min_df=2,
    max_df=0.90,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=3000
)
X = vectorizer.fit_transform(eligible["text_clean"])
feature_names = vectorizer.get_feature_names_out()
print("Comments x features:", X.shape)
print("Example features:", list(feature_names[:15]))
print("Custom excluded words remaining as features:",
      sorted(set(CUSTOM_STOP_WORDS).intersection(feature_names)))
# Comments containing only removed words/no retained features become all-zero vectors.
# Exclude them from K-Means but keep them in df for sampling diagnostics.
nonzero_mask = X.getnnz(axis=1) > 0
zero_vector_count = int((~nonzero_mask).sum())
print("Additional comments with zero TF-IDF features:", zero_vector_count)
if zero_vector_count:
    eligible = eligible.loc[nonzero_mask].reset_index(drop=True)
    X = X[nonzero_mask]
print("Comments remaining for K-Means:", X.shape[0])
if X.shape[0] < 12:
    raise ValueError("Too few comments remaining after stop-word/TF-IDF filtering.")
if X.shape[1] < 2:
    raise ValueError("Not enough distinct TF-IDF features.")


# Step 5 — Make the elbow plot

k_values = list(range(1, min(10, len(eligible) - 1) + 1))
inertia = []
silhouette = []
for k in k_values:
    trial = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    trial_labels = trial.fit_predict(X)
    inertia.append(trial.inertia_)
    if k >= 2 and len(set(trial_labels)) >= 2:
        value = silhouette_score(
            X, trial_labels, metric="cosine",
            sample_size=min(400, X.shape[0]), random_state=RANDOM_SEED
        )
    else:
        value = np.nan
    silhouette.append(value)
scores = pd.DataFrame({"k": k_values, "inertia": inertia,
                       "cosine_silhouette": silhouette})
print(scores.round(3).to_string(index=False))
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertia, marker="o", linestyle="--")
plt.xticks(k_values)
plt.xlabel("Number of clusters, k")
plt.ylabel("Inertia")
plt.title("Elbow method: midterm comments")
plt.grid(alpha=0.2)
plt.show()


# Step 6 — Choose k, fit K-Means and attach labels

CHOSEN_K = 4   # Change this after reading the plot and cluster contents.
final_model = KMeans(n_clusters=CHOSEN_K, random_state=RANDOM_SEED, n_init=10)
cluster_labels = final_model.fit_predict(X)
eligible["cluster"] = cluster_labels   # Like the class assign-back-to-dataframe example.
print("Final inertia:", round(final_model.inertia_, 2))
print("Cluster sizes:")
print(eligible["cluster"].value_counts().sort_index().to_string())


# Step 7 — Examine centroids, like the Iris exercise

centroids = final_model.cluster_centers_
rows = []
for cluster_id in range(CHOSEN_K):
    indices = np.argsort(centroids[cluster_id])[::-1][:10]
    rows.append({
        "cluster": cluster_id,
        "n_comments": int((eligible["cluster"] == cluster_id).sum()),
        "top_words": ", ".join(feature_names[i] for i in indices)
    })
centroid_table = pd.DataFrame(rows)
print(centroid_table.to_string(index=False))


# Step 8 — Inspect examples privately

SHOW_PRIVATE_EXAMPLES = False
distances = final_model.transform(X)
for cluster_id in range(CHOSEN_K):
    members = np.where(cluster_labels == cluster_id)[0]
    nearest = sorted(members, key=lambda i: distances[i, cluster_id])[:5]
    print(f"\nCluster {cluster_id}: {len(members)} eligible comments")
    print(centroid_table.loc[cluster_id, "top_words"])
    if SHOW_PRIVATE_EXAMPLES:
        for row_index in nearest:
            item = eligible.iloc[row_index]
            print(f"  {item['channel']} / {item['phase']}: {item['text'][:180]}")
    else:
        print("  Examples hidden; enable private review to assess interpretability.")


# Step 9 — Compare group percentages rather than raw counts

channels = ["HasanAbi", "Fox News"]
phases = ["before", "election", "after"]
index = pd.MultiIndex.from_product([channels, phases], names=["channel", "phase"])
counts = pd.crosstab(
    index=[eligible["channel"], eligible["phase"]],
    columns=eligible["cluster"]
).reindex(index, fill_value=0).reindex(columns=range(CHOSEN_K), fill_value=0)
percentages = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
print("Percent of eligible comments per group:")
print(percentages.round(1).to_string())
plt.figure(figsize=(9, 5))
plt.imshow(percentages.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
plt.colorbar(label="% of eligible group")
plt.xticks(range(CHOSEN_K), [f"Cluster {i}" for i in range(CHOSEN_K)])
plt.yticks(range(len(index)), [f"{c} / {p}" for c, p in index])
plt.title("How cluster prevalence changes over the election")
plt.tight_layout()
plt.show()


# Step 10 — Plot two dimensions, as in the Iris demo

svd = TruncatedSVD(n_components=2, random_state=RANDOM_SEED)
points_2d = svd.fit_transform(X)
centers_2d = svd.transform(final_model.cluster_centers_)
plt.figure(figsize=(8, 6))
plt.scatter(points_2d[:, 0], points_2d[:, 1],
            c=cluster_labels, s=20, alpha=0.5, cmap="viridis")
plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
            marker="X", c="red", s=150, label="Projected centroids")
plt.title("Comment clusters in a 2D projection")
plt.xlabel("SVD axis 1")
plt.ylabel("SVD axis 2")
plt.legend()
plt.show()
print("Fraction of variance explained by 2 axes:",
      f"{svd.explained_variance_ratio_.sum():.1%}")


# Step 11 — Repeat with other cluster counts

comparison = []
for k in [3, 4, 5, 6]:
    if k >= len(eligible):
        continue
    trial = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    assignments = trial.fit_predict(X)
    sizes = pd.Series(assignments).value_counts()
    comparison.append({"k": k, "inertia": round(trial.inertia_, 2),
                       "largest_cluster_pct": round(sizes.max() / len(eligible) * 100, 1),
                       "smallest_cluster_n": int(sizes.min())})
print(pd.DataFrame(comparison).to_string(index=False))


# Step 12 — Interpretation / reflection (write this in your own words)