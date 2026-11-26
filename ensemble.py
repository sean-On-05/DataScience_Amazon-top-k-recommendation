"""Measure candidate complementarity and test weighted reciprocal rank fusion."""

import argparse
import csv
from pathlib import Path

import numpy as np

import ease
import item_cf

CANDIDATES = 100
TOP_K = item_cf.TOP_K
CF_WEIGHTS = (0.1, 0.25, 0.5, 0.75, 0.9)
RANK_OFFSET = 20


def paired_candidates(path, users, shape):
    interactions = ease.interactions_from(path, shape)
    weights = ease.fit_ease(interactions, 500)
    ease_rows = ease.recommend(interactions, users, weights, top_k=CANDIDATES)
    cf_rows = item_cf.recommend(
        item_cf.read_pairs(path), users, shape[1], shape[0],
        shrinkage=5, top_k=CANDIDATES,
    )
    for (user, ease_items), (_, cf_items) in zip(ease_rows, cf_rows, strict=True):
        yield user, ease_items, cf_items


def rank_components(ease_items, cf_items):
    items, inverse = np.unique(
        np.concatenate((ease_items, cf_items)), return_inverse=True
    )
    a = np.zeros(len(items))
    b = np.zeros(len(items))
    a[inverse[:len(ease_items)]] = 1 / (RANK_OFFSET + np.arange(1, len(ease_items) + 1))
    b[inverse[len(ease_items):]] = 1 / (RANK_OFFSET + np.arange(1, len(cf_items) + 1))
    return items, a, b


def fuse(items, a, b, weight):
    score = (1 - weight) * a + weight * b
    return items[np.lexsort((items, -score))[:TOP_K]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ensemble_submission.csv"))
    args = parser.parse_args()
    shape = tuple(int(n) + 1 for n in item_cf.read_pairs(args.data / "train.csv").max(axis=0))
    truth = {}
    for user, item in item_cf.read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    candidate_denominator = sum(min(CANDIDATES, len(items)) for items in truth.values())
    union_denominator = sum(len(items) for items in truth.values())
    overlaps = {k: np.zeros(3, dtype=int) for k in (TOP_K, CANDIDATES)}
    hits = dict.fromkeys(CF_WEIGHTS, 0)

    for user, a, b in paired_candidates(
        args.validation / "validation_train.csv", sorted(truth), shape
    ):
        target = truth[user]
        for k, counts in overlaps.items():
            a_hits = target.intersection(a[:k])
            b_hits = target.intersection(b[:k])
            counts += (len(a_hits - b_hits), len(b_hits - a_hits), len(a_hits & b_hits))
        parts = rank_components(a, b)
        for weight in hits:
            hits[weight] += len(target.intersection(fuse(*parts, weight)))

    for k, (a_only, b_only, both) in overlaps.items():
        union = a_only + b_only + both
        print(f"Top {k}: EASE only={a_only}, CF only={b_only}, both={both}, union={union}")
        if k == CANDIDATES:
            print(f"Recall@100: EASE={(a_only + both) / candidate_denominator:.9f}, "
                  f"CF={(b_only + both) / candidate_denominator:.9f}")
            print(f"Union candidate coverage: {union}/{union_denominator} = {union / union_denominator:.9f}")
    for weight, count in hits.items():
        print(f"CF weight {weight:g}: {count}/{denominator} = {count / denominator:.9f}")
    best = max(hits, key=hits.get)
    ease_hits = overlaps[TOP_K][0] + overlaps[TOP_K][2]
    if hits[best] <= ease_hits:
        print("EASE retained; no ensemble submission generated.")
        return
    print(f"Selected CF weight {best:g}")
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test_users = [int(row["user_id"]) for row in csv.DictReader(source)]
    if sorted(users) != sorted(test_users) or len(set(users)) != len(users):
        raise ValueError("Sample submission and test user lists differ.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, a, b in paired_candidates(args.data / "train.csv", users, shape):
            writer.writerow([user, " ".join(map(str, fuse(*rank_components(a, b), best)))])


if __name__ == "__main__":
    main()
