"""Compare title-only and title-plus-category recommendations."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from item_cf import TOP_K, read_pairs


def category_names(value):
    if isinstance(value, str):
        if value.strip():
            yield value.strip()
    elif isinstance(value, list):
        for entry in value:
            yield from category_names(entry)


def load_metadata(path):
    metadata = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            metadata[int(record["item_id"])] = (
                (record.get("title") or "").strip(),
                list(dict.fromkeys(category_names(record.get("categories")))),
            )
    return metadata


def evaluate(texts, interactions, truth):
    features = TfidfVectorizer(dtype=np.float32).fit_transform(texts)
    counts = np.asarray(interactions.sum(axis=0)).ravel()
    item_indices = np.arange(len(texts))
    popularity = np.lexsort((item_indices, -counts))
    users = sorted(truth)
    hits = 0
    for start in range(0, len(users), 128):
        batch = users[start:start + 128]
        histories = interactions[batch]
        # Normalizing the sum is equivalent to normalizing the mean.
        profiles = normalize(histories @ features)
        scores = (profiles @ features.T).toarray()
        for index, user in enumerate(batch):
            row = scores[index]
            row[histories[index].indices] = -np.inf
            candidates = np.flatnonzero(row > 0)
            order = np.lexsort((candidates, -row[candidates]))
            selected = candidates[order[:TOP_K]].tolist()
            if len(selected) < TOP_K:
                for item in popularity:
                    if np.isfinite(row[item]) and item not in selected:
                        selected.append(int(item))
                    if len(selected) == TOP_K:
                        break
            hits += len(truth[user].intersection(selected))
    denominator = sum(min(len(items), TOP_K) for items in truth.values())
    return hits, denominator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, default=Path("data/item_meta.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    metadata = load_metadata(args.meta)
    training = read_pairs(args.validation / "validation_train.csv")
    validation = read_pairs(args.validation / "validation_truth.csv")
    catalog = sorted(metadata.keys() | set(training[:, 1]))
    indices = {item: index for index, item in enumerate(catalog)}
    truth = {}
    for user, item in validation:
        truth.setdefault(user, set()).add(indices.get(item, -int(item) - 1))
    interactions = csr_matrix(
        (np.ones(len(training), dtype=np.float32),
         (training[:, 0], [indices[item] for item in training[:, 1]])),
        shape=(max(int(training[:, 0].max()), max(truth)) + 1, len(catalog)),
    )
    interactions.data.fill(1)
    records = [metadata.get(item, ("", [])) for item in catalog]
    print(f"Catalog size: {len(catalog)}; metadata rows: {len(metadata)}")
    print(f"Missing titles: {sum(not title for title, _ in records)}")
    print(f"Missing categories: {sum(not categories for _, categories in records)}")
    print("Most common categories:", Counter(
        category for _, categories in records for category in categories
    ).most_common(5))
    for include_categories, label in ((False, "title"), (True, "title + categories")):
        texts = [
            " ".join([title, *categories]) if include_categories else title
            for title, categories in records
        ]
        hits, denominator = evaluate(texts, interactions, truth)
        print(f"{label} Recall@20: {hits}/{denominator} = {hits / denominator:.9f}")


if __name__ == "__main__":
    main()
