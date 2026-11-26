"""Evaluate EASE on the common split and build a Kaggle submission."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import lapack
from scipy.sparse import csr_matrix

from item_cf import TOP_K, read_pairs


def interactions_from(path, shape):
    pairs = read_pairs(path)
    interactions = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
        shape=shape,
    )
    interactions.data.fill(1)
    return interactions


def fit_ease(interactions, regularization):
    gram = (interactions.T @ interactions).toarray(order="F")
    diagonal = np.diag_indices_from(gram)
    gram[diagonal] += regularization
    precision, info = lapack.spotrf(gram, lower=True, overwrite_a=True, clean=False)
    if info:
        raise RuntimeError(f"EASE Cholesky factorization failed ({info}).")
    precision, info = lapack.spotri(precision, lower=True, overwrite_c=True)
    if info:
        raise RuntimeError(f"EASE precision calculation failed ({info}).")
    for column in range(len(precision)):
        precision[column, column + 1:] = precision[column + 1:, column]
    precision /= -precision[diagonal]
    precision[diagonal] = 0
    return np.ascontiguousarray(precision)


def recommend(interactions, users, weights):
    available = np.asarray(interactions.getnnz(axis=0)) > 0
    item_ids = np.arange(weights.shape[0])
    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        histories = interactions[batch]
        scores = histories @ weights
        scores[:, ~available] = -np.inf
        for index, user in enumerate(batch):
            seen = histories.indices[histories.indptr[index]:histories.indptr[index + 1]]
            row = scores[index]
            row[seen] = -np.inf
            cutoff = np.partition(row, -TOP_K)[-TOP_K]
            if not np.isfinite(cutoff):
                raise ValueError(f"Fewer than {TOP_K} eligible items for user {user}.")
            above = item_ids[row > cutoff]
            tied = item_ids[row == cutoff][:TOP_K - len(above)]
            candidates = np.concatenate((above, tied))
            yield user, candidates[np.lexsort((candidates, -row[candidates]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--regularization", type=float, nargs="+", default=[50, 200, 500])
    parser.add_argument("--output", type=Path, default=Path("outputs/ease_submission.csv"))
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in args.regularization):
        parser.error("Regularization values must be finite and positive.")

    full_pairs = read_pairs(args.data / "train.csv")
    shape = tuple(int(value) + 1 for value in full_pairs.max(axis=0))
    del full_pairs
    training = interactions_from(args.validation / "validation_train.csv", shape)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    users = sorted(truth)
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    results = []
    for regularization in args.regularization:
        weights = fit_ease(training, regularization)
        hits = sum(len(truth[user].intersection(items)) for user, items in
                   recommend(training, users, weights))
        del weights
        results.append((hits, regularization))
        print(f"lambda={regularization:g} Recall@20: {hits}/{denominator} = {hits / denominator:.9f}", flush=True)
    if args.validation_only:
        return

    _, best = max(results, key=lambda result: result[0])
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test_users = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test_users):
        raise ValueError("Sample submission and test user lists differ.")
    full = interactions_from(args.data / "train.csv", shape)
    weights = fit_ease(full, best)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, items in recommend(full, users, weights):
            writer.writerow([user, " ".join(map(str, items))])
    print(f"Selected lambda={best:g}; wrote {args.output}")


if __name__ == "__main__":
    main()
