"""Evaluate file-order neighbors with EASE using only competition interactions."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ease import fit_ease, interactions_from
from item_cf import TOP_K, read_pairs

WINDOWS = (10, 30, 100)
WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def read_review_order(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return np.asarray(
            [(int(row["user_id"]), int(row["item_id"])) for row in map(json.loads, source)],
            dtype=np.int64,
        )


def first_seen_order(raw_pairs, interactions):
    # Filter out validation pairs before taking each user's first appearance.
    retained = np.asarray(interactions[raw_pairs[:, 0], raw_pairs[:, 1]]).ravel() > 0
    users = raw_pairs[retained, 0]
    _, first = np.unique(users, return_index=True)
    return users[np.sort(first)]


def nearby_users(order, user_count, window):
    distances = np.r_[np.arange(-window, 0), np.arange(1, window + 1)]
    positions = np.arange(len(order))[:, None] + distances
    rows, offsets = np.nonzero((positions >= 0) & (positions < len(order)))
    return csr_matrix(
        ((1 / np.sqrt(np.abs(distances[offsets]))).astype(np.float32),
         (order[rows], order[positions[rows, offsets]])),
        shape=(user_count, user_count),
    )


def ranked_items(scores):
    threshold = np.partition(scores, -TOP_K)[-TOP_K]
    higher = np.flatnonzero(scores > threshold)
    ties = np.flatnonzero(scores == threshold)[:TOP_K - len(higher)]
    selected = np.concatenate((higher, ties))
    return selected[np.lexsort((selected, -scores[selected]))]


def normalize_scores(scores, histories, available):
    for row, (start, stop) in enumerate(zip(histories.indptr[:-1], histories.indptr[1:])):
        scores[row, histories.indices[start:stop]] = 0
    np.maximum(scores, 0, out=scores)
    scores[:, ~available] = 0
    scores /= np.maximum(scores.max(axis=1, keepdims=True), 1e-12)
    return scores


def evaluate(interactions, users, order, configs, shuffle_control=False):
    weights = fit_ease(interactions, 500)
    frequency = np.asarray(interactions.sum(axis=0)).ravel()
    available = frequency > 0
    inverse_popularity = 1 / np.sqrt(np.maximum(frequency, 1))
    neighborhoods = {
        window: nearby_users(order, interactions.shape[0], window)
        for window in sorted({window for window, _ in configs if window})
    }
    # Evaluate the shuffled control for every configured blend in the same run.
    shuffled = np.random.default_rng(42).permutation(order)
    controls = {
        window: nearby_users(shuffled, interactions.shape[0], window)
        for window in neighborhoods
    } if shuffle_control else {}

    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        histories = interactions[batch]
        raw_base = histories @ weights
        base = normalize_scores(raw_base.copy(), histories, available)
        local = {
            (kind, window): normalize_scores(
                (matrix[batch] @ interactions).toarray() * inverse_popularity,
                histories, available,
            )
            for kind, matrices in (("file", neighborhoods), ("shuffled", controls))
            for window, matrix in matrices.items()
        }
        scores_by_config = {
            (kind, window, blend): raw_base if not window else
            (1 - blend) * base + blend * local[kind, window]
            for window, blend in configs
            for kind in (("file", "shuffled") if window and controls else ("file",))
        }
        for position, user in enumerate(batch):
            seen = histories.indices[histories.indptr[position]:histories.indptr[position + 1]]
            results = {}
            for config, scores in scores_by_config.items():
                row = scores[position]
                row[seen] = -np.inf
                row[~available] = -np.inf
                results[config] = ranked_items(row)
            yield user, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--review-order", type=Path, help="Competition train_reviews.jsonl or .jsonl.gz")
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/ordering_submission.csv"))
    args = parser.parse_args()
    raw = read_pairs(args.data / "train.csv")
    shape = tuple(int(value) + 1 for value in raw.max(axis=0))
    order_pairs = read_review_order(args.review_order) if args.review_order else raw
    if (len(order_pairs) != len(raw) or np.any(order_pairs < 0)
            or np.any(order_pairs >= shape) or not np.array_equal(
        np.sort(order_pairs[:, 0] * shape[1] + order_pairs[:, 1]),
        np.sort(raw[:, 0] * shape[1] + raw[:, 1]),
    )):
        raise ValueError("Review-order records must match the competition train.csv pairs.")
    train = interactions_from(args.validation / "validation_train.csv", shape)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    configs = [(0, 0)] + [(window, blend) for window in WINDOWS for blend in WEIGHTS]
    hits = {}
    order = first_seen_order(order_pairs, train)
    for user, results in evaluate(train, sorted(truth), order, configs, shuffle_control=True):
        for config, items in results.items():
            hits[config] = hits.get(config, 0) + len(truth[user].intersection(items))
    denominator = sum(min(len(items), TOP_K) for items in truth.values())
    for config, count in hits.items():
        print(f"{config}: {count}/{denominator} = {count / denominator:.9f}")
    best = max((config for config in hits if config[0] == "file"), key=hits.get)
    print(f"Selected {best}", flush=True)
    if args.validation_only or best[1] == 0:
        return

    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test):
        raise ValueError("Sample submission and test user lists differ.")
    full = interactions_from(args.data / "train.csv", shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, results in evaluate(full, users, first_seen_order(order_pairs, full), [best[1:]]):
            writer.writerow([user, " ".join(map(str, results[best]))])


if __name__ == "__main__":
    main()
