"""Diagnose held-out positives in the EASE + shrinkage CF recommendations."""

import argparse
from collections import Counter
from pathlib import Path

from ease import interactions_from
from ensemble import TOP_K, fuse, paired_candidates, rank_components
from item_cf import read_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    shape = tuple(int(n) + 1 for n in read_pairs(args.data / "train.csv").max(axis=0))
    fit = args.validation / "validation_train.csv"
    interactions = interactions_from(fit, shape)
    co_reviews = (interactions.T @ interactions).tocsr()
    truth = {}
    for user, item in read_pairs(args.validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    counts = Counter()
    support = Counter()
    oracle = 0
    for user, ease, cf in paired_candidates(fit, sorted(truth), shape):
        target = truth[user]
        ease_top = target.intersection(ease[:TOP_K])
        cf_top = target.intersection(cf[:TOP_K])
        candidates = target.intersection(ease).union(target.intersection(cf))
        oracle += min(TOP_K, len(candidates))
        selected = target.intersection(fuse(*rank_components(ease, cf), 0.5))
        groups = {
            "both top 20": ease_top & cf_top,
            "EASE-only top 20": ease_top - cf_top,
            "CF-only top 20": cf_top - ease_top,
            "candidate outside both top 20": candidates - ease_top - cf_top,
            "absent from both top 100": target - candidates,
        }
        for name, items in groups.items():
            counts[name, "total"] += len(items)
            counts[name, "kept"] += len(items & selected)
        history = interactions.indices[interactions.indptr[user]:interactions.indptr[user + 1]]
        for name in ("CF-only top 20", "candidate outside both top 20"):
            for item in groups[name]:
                links = co_reviews[history, item].toarray().ravel()
                peak = links.max(initial=0)
                bucket = "0–1" if peak <= 1 else "2–4" if peak <= 4 else "5+"
                support[name, bucket, "total"] += 1
                support[name, bucket, "kept"] += item in selected

    print("Group | Total | Kept | Missed")
    for name in groups:
        total = counts[name, "total"]
        kept = counts[name, "kept"]
        print(f"{name} | {total} | {kept} | {total - kept}")
    denominator = sum(min(TOP_K, len(items)) for items in truth.values())
    print(f"Oracle candidate Recall@20: {oracle}/{denominator} = {oracle / denominator:.9f}")
    print("Group | Max co-reviews | Total | Kept")
    for name in ("CF-only top 20", "candidate outside both top 20"):
        for bucket in ("0–1", "2–4", "5+"):
            print(f"{name} | {bucket} | {support[name, bucket, 'total']} | {support[name, bucket, 'kept']}")


if __name__ == "__main__":
    main()
