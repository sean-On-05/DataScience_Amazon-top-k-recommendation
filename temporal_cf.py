"""Evaluate timestamp-weighted item CF and user CF on the common split."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ease import interactions_from
from item_cf import TOP_K, read_pairs

DAY = 86_400_000
SETTINGS = ("temporal-7", "temporal-30", "temporal-90", "user-knn",
            "temporal-30+user", "temporal-7+30+90")


def load_timestamps(path, interactions):
    timestamps = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            key = int(row["user_id"]), int(row["item_id"])
            stamp = int(float(row["timestamp"]))
            timestamps[key] = max(stamp, timestamps.get(key, 0))
    return [
        np.array([timestamps[user, item] for item in interactions.indices[start:stop]],
                 dtype=np.float64)
        for user, (start, stop) in enumerate(zip(interactions.indptr[:-1], interactions.indptr[1:]))
    ]


def item_models(interactions, times):
    row_blocks, col_blocks, gap_blocks = [], [], []
    for user, (start, stop) in enumerate(zip(interactions.indptr[:-1], interactions.indptr[1:])):
        items = interactions.indices[start:stop]
        left, right = np.triu_indices(len(items), 1)
        row_blocks.append(items[left])
        col_blocks.append(items[right])
        gap_blocks.append((np.abs(times[user][left] - times[user][right]) / DAY).astype(np.float32))
    rows, cols, gaps = map(np.concatenate, (row_blocks, col_blocks, gap_blocks))
    counts = np.asarray(interactions.sum(axis=0)).ravel()
    scale = np.sqrt(np.maximum(counts, 1))
    matrices = []
    for days in (7, 30, 90):
        values = np.exp(-gaps / days) + 0.03
        upper = csr_matrix((values, (rows, cols)), shape=(len(counts), len(counts)))
        matrix = (upper + upper.T).tocoo()
        evidence = matrix.data
        matrix.data = evidence * (evidence / (evidence + 2))
        matrix.data /= scale[matrix.row] * scale[matrix.col]
        matrices.append(matrix.tocsr())
    return matrices


def user_scores(interactions, batch):
    histories = interactions[batch]
    similarities = (histories @ interactions.T).tocsr()
    size = np.sqrt(np.maximum(interactions.getnnz(axis=1), 1))
    for index, user in enumerate(batch):
        start, stop = similarities.indptr[index:index + 2]
        counts = similarities.data[start:stop]
        neighbors = similarities.indices[start:stop]
        counts[:] = counts ** 2 / ((counts + 3) * size[user] * size[neighbors])
        counts[neighbors == user] = 0
        if len(counts) > 100:
            order = np.lexsort((neighbors, -counts))
            counts[order[100:]] = 0
    similarities.eliminate_zeros()
    return (similarities @ interactions).toarray()


def top_items(row):
    cutoff = np.partition(row, -TOP_K)[-TOP_K]
    above = np.flatnonzero(row > cutoff)
    tied = np.flatnonzero(row == cutoff)[:TOP_K - len(above)]
    candidates = np.concatenate((above, tied))
    return candidates[np.lexsort((candidates, -row[candidates]))]


def recommend(interactions, times, users, settings):
    matrices = item_models(interactions, times)
    frequency = np.asarray(interactions.sum(axis=0)).ravel()
    popularity = frequency / max(frequency.max(), 1)
    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        histories = interactions[batch]
        scores = [(histories @ matrix).toarray() for matrix in matrices]
        scores.append(user_scores(interactions, batch))
        for values in scores:
            for index, (left, right) in enumerate(zip(histories.indptr[:-1], histories.indptr[1:])):
                values[index, histories.indices[left:right]] = 0
            values /= np.maximum(values.max(axis=1, keepdims=True), 1e-12)

        scores.extend(((scores[1] + scores[3]) / 2, sum(scores[:3]) / 3))
        for index, user in enumerate(batch):
            seen = histories.indices[histories.indptr[index]:histories.indptr[index + 1]]
            predictions = {}
            for name in settings:
                row = scores[SETTINGS.index(name)][index] + 1e-8 * popularity
                row[seen] = -np.inf
                row[frequency == 0] = -np.inf
                predictions[name] = top_items(row)
            yield user, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/temporal_cf_submission.csv"))
    args = parser.parse_args()
    train_file = args.data / "train.csv"
    shape = tuple(int(n) + 1 for n in read_pairs(train_file).max(axis=0))
    interactions = interactions_from(args.validation / "validation_train.csv", shape)
    times = load_timestamps(train_file, interactions)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    hits = dict.fromkeys(SETTINGS, 0)
    for user, predictions in recommend(interactions, times, sorted(truth), SETTINGS):
        for name, items in predictions.items():
            hits[name] += len(truth[user].intersection(items))
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    for name, count in hits.items():
        print(f"{name}: {count}/{denominator} = {count / denominator:.9f}")
    best = max(hits, key=hits.get)
    if hits[best] <= 2980:
        print("Existing ensemble retained.")
        return
    print(f"Selected {best}")
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test_users = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test_users):
        raise ValueError("Sample submission and test user lists differ.")
    full = interactions_from(train_file, shape)
    times = load_timestamps(train_file, full)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, predictions in recommend(full, times, users, (best,)):
            writer.writerow([user, " ".join(map(str, predictions[best]))])


if __name__ == "__main__":
    main()
