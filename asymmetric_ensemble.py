"""Compare rank fusion using directionally normalized item CF."""

import argparse
import csv
from pathlib import Path

import numpy as np

from asymmetric_cf import evaluate_rows
from ease import fit_ease, interactions_from, recommend
from ensemble import CANDIDATES, fuse, rank_components
from item_cf import TOP_K, read_pairs

WEIGHTS = (0.25, 0.5, 0.75)
CF_SETTING = (0.25, 5)


def components(interactions, users):
    weights = fit_ease(interactions, 500)
    ease_rows = recommend(interactions, users, weights, top_k=CANDIDATES)
    cf_rows = evaluate_rows(interactions, users, [CF_SETTING], top_k=CANDIDATES)
    for (user, ease_items), (_, (cf_items,)) in zip(ease_rows, cf_rows, strict=True):
        yield user, rank_components(ease_items, cf_items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/asymmetric_ensemble_submission.csv"))
    args = parser.parse_args()
    full_file = args.data / "train.csv"
    shape = tuple(int(n) + 1 for n in read_pairs(full_file).max(axis=0))
    train = interactions_from(args.validation / "validation_train.csv", shape)
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    hits = dict.fromkeys(WEIGHTS, 0)
    for user, parts in components(train, sorted(truth)):
        for weight in hits:
            hits[weight] += len(truth[user].intersection(fuse(*parts, weight)))
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    for weight, count in hits.items():
        print(f"Asymmetric CF weight {weight:g}: {count}/{denominator} = {count / denominator:.9f}")
    best = max(hits, key=hits.get)
    if hits[best] <= 2980:
        return
    print(f"Selected asymmetric CF weight {best:g}")
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test):
        raise ValueError("Sample submission and test user lists differ.")
    full = interactions_from(full_file, shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for user, parts in components(full, users):
            writer.writerow([user, " ".join(map(str, fuse(*parts, best)))])


if __name__ == "__main__":
    main()
