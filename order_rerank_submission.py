"""Compare learned reranking and rank fusion on disjoint user groups."""

import argparse
import csv
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from content_model import category_names
from ease import interactions_from
from ensemble import paired_candidates
from item_cf import read_pairs
from wide_rerank import train_submission_model
from ordering import read_review_order, first_seen_order, nearby_users
from lexical_candidates import best_indices, user_candidates

FIELDS = (
    "ease_rank", "cf_rank", "title_rank", "turbo_rank", "rrf3", "rrf4",
    "item_frequency", "history_size", "title_max", "title_mean", "title_second",
    "co_max", "co_sum", "co_links", "co_strong_links", "same_store",
    "same_leaf_category", "same_store_nearest", "same_leaf_nearest",
    "nearest_rating", "nearest_age_days", "mean_related_rating",
)


@contextmanager
def metadata_rows(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as source:
            name = next(n for n in source.namelist()
                        if Path(n).name == "item_meta.jsonl" and not n.startswith("__MACOSX/"))
            with source.open(name) as rows:
                yield rows
    else:
        with path.open(encoding="utf-8") as rows:
            yield rows


def catalog_features(item_count, metadata_path):
    titles = [""] * item_count
    stores = [""] * item_count
    leaves = [""] * item_count
    with metadata_rows(metadata_path) as rows:
        for row in map(json.loads, rows):
            item = int(row["item_id"])
            titles[item] = row.get("title") or ""
            stores[item] = str(row.get("store") or "").strip().casefold()
            categories = list(category_names(row.get("categories")))
            if len(categories) > 2:
                leaves[item] = categories[-1].casefold()

    word = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32).fit_transform(titles)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                            dtype=np.float32).fit_transform(titles)
    text = hstack((word, chars), format="csr")
    neighbors = np.empty((item_count, 100), dtype=np.int32)
    similarities = np.empty(neighbors.shape, dtype=np.float32)
    for start in range(0, item_count, 128):
        block = (text[start:start + 128] @ text.T).toarray() * 0.5
        for offset, scores in enumerate(block):
            scores[start + offset] = -np.inf
            indices = best_indices(scores, 100)
            neighbors[start + offset] = indices
            similarities[start + offset] = scores[indices]
    return text, neighbors, similarities, np.array(stores), np.array(leaves)


def graph(interactions, alpha, power):
    users = np.maximum(interactions.getnnz(axis=1), 1).astype(np.float32)
    items = np.maximum(interactions.getnnz(axis=0), 1).astype(np.float32)
    scaled = interactions.multiply(users[:, None] ** -alpha).multiply(items ** (alpha - 1)).tocsr()
    gram = (scaled.T @ scaled).tocsr()
    np.power(gram.data, power, out=gram.data)
    return gram



def graph_rows(interactions, users):
    matrix = graph(interactions, 0.7, 0.6)
    unavailable = interactions.getnnz(axis=0) == 0
    for start in range(0, len(users), 128):
        histories = interactions[users[start:start + 128]]
        arrays = (histories @ matrix).toarray()
        arrays[:, unavailable] = -np.inf
        for position, scores in enumerate(arrays):
            left, right = histories.indptr[position:position + 2]
            scores[histories.indices[left:right]] = -np.inf
            yield best_indices(scores, 100)


def candidate_features(full_path, split_path, metadata_path, users, reviews):
    shape = tuple(int(n) + 1 for n in read_pairs(full_path).max(axis=0))
    fit = interactions_from(split_path, shape)
    co = (fit.T @ fit).tocsr()
    frequency = fit.getnnz(axis=0)
    ratings = {}
    # Keep attributes only for interactions visible in the validation training split.
    retained = set(map(tuple, read_pairs(split_path)))
    with full_path.open() as source:
        for row in csv.DictReader(source):
            key = int(row["user_id"]), int(row["item_id"])
            if key in retained:
                ratings[key] = float(row["rating"]), float(row["timestamp"])
    del retained
    text, neighbors, sims, stores, leaves = catalog_features(shape[1], metadata_path)
    ordered = first_seen_order(read_review_order(reviews), fit)
    if len(ordered) != len(users):
        raise ValueError("Review order did not cover all validation users.")
    proximity = nearby_users(ordered, shape[0], 30)
    popularity_correction = 1 / np.sqrt(np.maximum(frequency, 1))

    def proximity_scores():
        for start in range(0, len(users), 128):
            batch = users[start:start+128]
            yield from (proximity[batch] @ fit).toarray() * popularity_correction

    print("Title neighbors and review-order graph ready.", flush=True)
    for (user, ease, cf), turbo, order_scores in zip(
        paired_candidates(split_path, users, shape), graph_rows(fit, users),
        proximity_scores(), strict=True
    ):
        left, right = fit.indptr[user:user + 2]
        history = fit.indices[left:right]
        title = user_candidates(history, neighbors, sims, shape[1])
        order_scores[history] = 0
        order_candidates = best_indices(order_scores, 100)
        order_candidates = order_candidates[order_scores[order_candidates] > 0]
        items = np.unique(np.concatenate((ease, cf, title, turbo, order_candidates)))
        order_rank = np.zeros(len(items), dtype=np.float32)
        order_rank[np.searchsorted(items, order_candidates)] = 1 / (21 + np.arange(len(order_candidates)))
        old_candidate = np.isin(items, np.concatenate((ease, cf, title, turbo)))
        reciprocal = np.zeros((len(items), 4), dtype=np.float32)
        for col, ranked in enumerate((ease, cf, title, turbo)):
            reciprocal[np.searchsorted(items, ranked), col] = 1 / (21 + np.arange(len(ranked)))
        rrf3 = reciprocal[:, :3].mean(axis=1)
        rrf4 = 0.9 * rrf3 + 0.1 * reciprocal[:, 3]
        title_sim = (text[items] @ text[history].T).toarray() * 0.5
        nearest = title_sim.argmax(axis=1)
        title_max = title_sim[np.arange(len(items)), nearest]
        second = np.partition(title_sim, -2, axis=1)[:, -2] if len(history) > 1 else title_max
        joint = co[items][:, history].toarray()
        review = np.array([ratings[user, item] for item in history])
        age = (review[:, 1].max() - review[:, 1]) / 86_400_000
        same_store = (stores[items, None] == stores[history]) & (stores[items, None] != "")
        same_leaf = (leaves[items, None] == leaves[history]) & (leaves[items, None] != "")
        blocks = (
            reciprocal, rrf3, rrf4, np.log1p(frequency[items]), np.full(len(items), len(history)),
            title_max, title_sim.mean(axis=1), second, joint.max(axis=1),
            joint.sum(axis=1), (joint > 0).sum(axis=1), (joint >= 5).sum(axis=1),
            same_store.sum(axis=1), same_leaf.sum(axis=1),
            same_store[np.arange(len(items)), nearest], same_leaf[np.arange(len(items)), nearest],
            review[nearest, 0], np.log1p(age[nearest]),
            (joint @ review[:, 0]) / np.maximum(joint.sum(axis=1), 1),
        )
        features = np.column_stack((*blocks, order_rank, order_scores[items], old_candidate))
        yield user, items, features.astype(np.float32)


def generate(full_path, validation, metadata_path, reviews):
    truth = {}
    for user, item in read_pairs(validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    users = sorted(truth)
    feature_blocks = []
    label_blocks = []
    item_blocks = []
    offsets = [0]
    for user, items, features in candidate_features(
        full_path, validation / "validation_train.csv", metadata_path, users, reviews
    ):
        feature_blocks.append(features)
        label_blocks.append(np.isin(items, list(truth[user])))
        item_blocks.append(items.astype(np.int32))
        offsets.append(offsets[-1] + len(items))
        if len(offsets) % 8000 == 0:
            print(f"Generated features for {len(offsets) - 1} users.", flush=True)

    features = np.concatenate(feature_blocks)
    labels = np.concatenate(label_blocks)
    items = np.concatenate(item_blocks)
    denominators = np.array([min(20, len(truth[user])) for user in users])
    return features, labels, items, np.array(offsets), denominators



def main():
    parser = argparse.ArgumentParser(description="Build the review-order Kaggle submission.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--meta", type=Path, default=Path("data/item_meta.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("data/train_reviews.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/order_rerank_submission.csv"))
    args = parser.parse_args()
    train = args.data / "train.csv"
    features, labels, _, _, _ = generate(train, args.validation, args.meta, args.reviews)
    model = train_submission_model(features, labels)
    del features, labels
    print("Trained order-aware reranker.", flush=True)
    with (args.data / "sample_submission.csv").open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with (args.data / "test.csv").open(newline="") as source:
        test = [int(row["user_id"]) for row in csv.DictReader(source)]
    if sorted(users) != sorted(test) or len(set(users)) != len(users):
        raise ValueError("Sample submission and test user lists differ.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".tmp")
    with partial.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        for index, (user, items, values) in enumerate(
            candidate_features(train, train, args.meta, users, args.reviews), 1
        ):
            blend = np.zeros(len(items))
            for weight, scores in ((.75, model.predict_proba(values)[:, 1]), (.25, values[:, 4])):
                order = np.lexsort((items, -scores))
                blend[order] += weight / (21 + np.arange(len(order)))
            top = items[np.lexsort((items, -blend))[:20]]
            writer.writerow([user, " ".join(map(str, top))])
            if index % 8000 == 0:
                print(f"Ranked {index} submission users.", flush=True)
    partial.replace(args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
