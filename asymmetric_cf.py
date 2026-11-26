"""Test directional co-review scoring and produce a Kaggle CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np

from ease import interactions_from
from item_cf import TOP_K, read_pairs

SETTINGS = ((0.25, 5), (0.0, 5), (0.15, 20), (0.30, 20), (0.50, 20), (0.15, 100))
BASELINE_HITS = 2980


def evaluate_rows(interactions, users, settings, *, top_k=TOP_K):
    counts = np.asarray(interactions.sum(axis=0)).ravel()
    safe_counts = np.maximum(counts, 1)
    base = (interactions.T @ interactions).tocsr()
    base.setdiag(0)
    base.eliminate_zeros()
    co = base.tocoo()
    matrices = []
    for target_power, shrinkage in settings:
        matrix = base.copy()
        matrix.data = (co.data ** 2 / (co.data + shrinkage)
                       / safe_counts[co.row] ** 0.5 / safe_counts[co.col] ** target_power)
        matrices.append(matrix)

    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        histories = interactions[batch]
        all_scores = [(histories @ matrix).toarray() for matrix in matrices]
        for index, user in enumerate(batch):
            seen = histories.indices[histories.indptr[index]:histories.indptr[index + 1]]
            rankings = []
            for scores in all_scores:
                row = scores[index]
                row[seen] = -np.inf
                row[counts == 0] = -np.inf
                cutoff = np.partition(row, -top_k)[-top_k]
                above = np.flatnonzero(row > cutoff)
                tied = np.flatnonzero(row == cutoff)[:top_k - len(above)]
                items = np.concatenate((above, tied))
                rankings.append(items[np.lexsort((items, -row[items]))])
            yield user, rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/asymmetric_cf_submission.csv"))
    args = parser.parse_args()
    path = args.data / "train.csv"
    shape = tuple(int(n) + 1 for n in read_pairs(path).max(axis=0))
    train = interactions_from(args.validation / "validation_train.csv", shape)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    hits = np.zeros(len(SETTINGS), dtype=int)
    for user, rankings in evaluate_rows(train, sorted(truth), SETTINGS):
        hits += [len(truth[user].intersection(items)) for items in rankings]
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    for config, count in zip(SETTINGS, hits):
        print(f"{config}: {count}/{denominator} = {count / denominator:.9f}")
    best = int(hits.argmax())
    if hits[best] <= BASELINE_HITS:
        return
    print(f"Selected {SETTINGS[best]}")
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test):
        raise ValueError("Sample submission and test user lists differ.")
    full = interactions_from(path, shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, (items,) in evaluate_rows(full, users, [SETTINGS[best]]):
            writer.writerow([user, " ".join(map(str, items))])


if __name__ == "__main__":
    main()
