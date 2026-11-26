"""Evaluate item-based collaborative filtering and generate a Kaggle CSV."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

TOP_K = 20
BATCH_SIZE = 256


def read_pairs(path):
    with path.open(newline="") as source:
        return np.asarray(
            [(int(row["user_id"]), int(row["item_id"])) for row in csv.DictReader(source)],
            dtype=np.int64,
        )


def recommend(pairs, users, item_count, user_count):
    interactions = csr_matrix(
        (np.ones(len(pairs), dtype=np.float32), (pairs[:, 0], pairs[:, 1])),
        shape=(user_count, item_count),
    )
    interactions.data.fill(1)
    item_frequency = np.asarray(interactions.sum(axis=0)).ravel()
    available = item_frequency > 0
    popularity = np.flatnonzero(available)
    popularity = popularity[np.lexsort((popularity, -item_frequency[popularity]))]
    scale = np.sqrt(np.maximum(item_frequency, 1))
    similarity = (interactions.T @ interactions).tocoo()
    similarity.data /= scale[similarity.row] * scale[similarity.col]
    similarity = similarity.tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()

    for start in range(0, len(users), BATCH_SIZE):
        batch_users = users[start : start + BATCH_SIZE]
        scores = (interactions[batch_users] @ similarity).tocsr()
        for index, user in enumerate(batch_users):
            seen = set(interactions[user].indices)
            row = scores[index]
            eligible = np.array([item not in seen for item in row.indices], dtype=bool)
            candidates, values = row.indices[eligible], row.data[eligible]
            order = np.lexsort((candidates, -values))[:TOP_K]
            selected = candidates[order].tolist()
            if len(selected) < TOP_K:
                for item in popularity:
                    if item not in seen and item not in selected:
                        selected.append(int(item))
                    if len(selected) == TOP_K:
                        break
            if len(selected) != TOP_K:
                raise ValueError(f"Fewer than {TOP_K} eligible items for user {user}")
            yield user, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/item_cf_submission.csv"))
    args = parser.parse_args()

    full = read_pairs(args.data / "train.csv")
    training = read_pairs(args.validation / "validation_train.csv")
    validation = read_pairs(args.validation / "validation_truth.csv")
    with (args.data / "test.csv").open(newline="") as source:
        test_users = np.array([int(row["user_id"]) for row in csv.DictReader(source)])
    with (args.data / "sample_submission.csv").open(newline="") as source:
        sample_users = np.array([int(row["user_id"]) for row in csv.DictReader(source)])
    if len(np.unique(sample_users)) != len(sample_users) or not np.array_equal(
        np.sort(sample_users), np.sort(test_users)
    ):
        raise ValueError("Sample submission and test user lists differ.")

    item_count = int(full[:, 1].max()) + 1
    user_count = int(full[:, 0].max()) + 1
    truth = {}
    for user, item in validation:
        truth.setdefault(user, set()).add(item)
    hits = sum(
        len(truth[user].intersection(items))
        for user, items in recommend(training, np.array(sorted(truth)), item_count, user_count)
    )
    denominator = sum(min(len(items), TOP_K) for items in truth.values())
    print(f"Validation Recall@20: {hits}/{denominator} = {hits / denominator:.9f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, items in recommend(full, sample_users, item_count, user_count):
            writer.writerow([user, " ".join(map(str, items))])


if __name__ == "__main__":
    main()
