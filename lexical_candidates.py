"""Test title-based nearest-item retrieval on the existing validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from ease import interactions_from
from ensemble import CANDIDATES, RANK_OFFSET, TOP_K, fuse, paired_candidates, rank_components
from item_cf import read_pairs

WEIGHTS = (0.05, 0.10, 0.20, 1 / 3)


def best_indices(scores, k):
    cutoff = np.partition(scores, -k)[-k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - len(above)]
    indices = np.concatenate((above, tied))
    return indices[np.lexsort((indices, -scores[indices]))]


def title_neighbors(titles):
    word = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32).fit_transform(titles)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           dtype=np.float32).fit_transform(titles)
    features = hstack((word, char), format="csr")
    indices = np.empty((len(titles), CANDIDATES), dtype=np.int32)
    values = np.empty(indices.shape, dtype=np.float32)
    for start in range(0, len(titles), 128):
        similarity = (features[start:start + 128] @ features.T).toarray()
        similarity *= 0.5
        for offset, row in enumerate(similarity):
            item = start + offset
            row[item] = -np.inf
            top = best_indices(row, CANDIDATES)
            indices[item], values[item] = top, row[top]
    return indices, values


def user_candidates(history, neighbors, similarities, item_count):
    scores = np.zeros(item_count, dtype=np.float32)
    np.maximum.at(scores, neighbors[history].ravel(), similarities[history].ravel())
    scores[history] = 0
    candidates = best_indices(scores, CANDIDATES)
    return candidates[scores[candidates] > 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/item_meta.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    train = args.validation / "validation_train.csv"
    full = args.train
    shape = tuple(int(n) + 1 for n in read_pairs(full).max(axis=0))
    titles = [""] * shape[1]
    with args.meta.open(encoding="utf-8") as source:
        for row in map(json.loads, source):
            titles[int(row["item_id"])] = row.get("title") or ""
    neighbors, similarities = title_neighbors(titles)
    print("Built item-level title candidates.", flush=True)
    interactions = interactions_from(train, shape)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    hits = dict.fromkeys(WEIGHTS, 0)
    bucket_counts = dict.fromkeys(("base_20", "title_20", "title_100", "base_union",
                                    "new_truth", "expanded_union", "oracle_20"), 0)
    for user, a, b in paired_candidates(train, sorted(truth), shape):
        target = truth[user]
        history = interactions.indices[interactions.indptr[user]:interactions.indptr[user + 1]]
        c = user_candidates(history, neighbors, similarities, shape[1])
        old, rank_a, rank_b = rank_components(a, b)
        original_coverage = target.intersection(old)
        title_coverage = target.intersection(c)
        expanded = original_coverage | title_coverage
        bucket_counts["base_20"] += len(target.intersection(fuse(old, rank_a, rank_b, 0.5)))
        bucket_counts["title_20"] += len(target.intersection(c[:TOP_K]))
        bucket_counts["title_100"] += len(title_coverage)
        bucket_counts["base_union"] += len(original_coverage)
        bucket_counts["new_truth"] += len(title_coverage - original_coverage)
        bucket_counts["expanded_union"] += len(expanded)
        bucket_counts["oracle_20"] += min(TOP_K, len(expanded))

        union, inverse = np.unique(np.concatenate((old, c)), return_inverse=True)
        base = np.zeros(len(union))
        title = np.zeros(len(union))
        base[inverse[:len(old)]] = (rank_a + rank_b) / 2
        title[inverse[len(old):]] = 1 / (RANK_OFFSET + np.arange(1, len(c) + 1))
        for weight in hits:
            hits[weight] += len(target.intersection(fuse(union, base, title, weight)))

    print(bucket_counts)
    denom = sum(min(TOP_K, len(items)) for items in truth.values())
    total = sum(map(len, truth.values()))
    print(f"Coverage: {bucket_counts['expanded_union']}/{total}")
    print(f"Title-only Recall@20: {bucket_counts['title_20']}/{denom} = {bucket_counts['title_20'] / denom:.9f}")
    for weight, n in hits.items():
        print(f"Title weight {weight:.6f}: {n}/{denom} = {n / denom:.9f}")


if __name__ == "__main__":
    main()
