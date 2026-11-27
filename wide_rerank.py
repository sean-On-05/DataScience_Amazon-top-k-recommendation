"""Compare learned reranking and rank fusion on disjoint user groups."""

import argparse
import csv
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from content_model import category_names
from ease import interactions_from
from ensemble import paired_candidates
from item_cf import read_pairs
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


def candidate_features(full_path, split_path, metadata_path, users):
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
    print("Title, store and category features ready.", flush=True)

    for (user, ease, cf), turbo in zip(
        paired_candidates(split_path, users, shape), graph_rows(fit, users), strict=True
    ):
        left, right = fit.indptr[user:user + 2]
        history = fit.indices[left:right]
        title = user_candidates(history, neighbors, sims, shape[1])
        items = np.unique(np.concatenate((ease, cf, title, turbo)))
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
        yield user, items, np.column_stack(blocks).astype(np.float32)


def generate(full_path, validation, metadata_path):
    truth = {}
    for user, item in read_pairs(validation / "validation_truth.csv"):
        truth.setdefault(user, set()).add(item)
    users = sorted(truth)
    feature_blocks = []
    label_blocks = []
    item_blocks = []
    offsets = [0]
    for user, items, features in candidate_features(
        full_path, validation / "validation_train.csv", metadata_path, users
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


def evaluate(features, labels, items, offsets, denominators):
    rng = np.random.default_rng(2026)
    user_groups = np.array_split(rng.permutation(len(denominators)), [int(len(denominators) * 0.75)])
    learn_users, eval_users = user_groups
    train_rows = np.concatenate([np.arange(offsets[u], offsets[u + 1]) for u in learn_users])
    positive_rows = train_rows[labels[train_rows]]
    negative_rows = train_rows[~labels[train_rows]]
    negative_rows = rng.choice(negative_rows, size=min(len(negative_rows), 650000), replace=False)
    selected = np.concatenate((positive_rows, negative_rows))
    eval_users = np.sort(eval_users)
    eval_rows = np.concatenate([np.arange(offsets[u], offsets[u + 1]) for u in eval_users])
    x = features[eval_rows]
    total = int(denominators[eval_users].sum())
    print(f"Training users={len(learn_users)}, held-out users={len(eval_users)}, "
          f"train positives={len(positive_rows)}, denominator={total}", flush=True)
    scores = {"RRF 3": x[:, 4].astype(np.float64), "RRF 4": x[:, 5].astype(np.float64)}
    for name, columns in (("ranks + popularity", np.r_[0:8]), ("all features", np.arange(len(FIELDS)))):
        model = fit_reranker(features[selected][:, columns], labels[selected])
        scores[name] = model.predict_proba(x[:, columns])[:, 1]

    hits = dict.fromkeys(scores, 0)
    hits.update({"all features + RRF 3 (25%)": 0, "all features + RRF 3 (50%)": 0})
    pointer = coverage = 0
    for u in eval_users:
        left, right = offsets[u:u + 2]
        section = slice(pointer, pointer + right - left)
        candidates = items[left:right]
        targets = labels[left:right]
        coverage += min(20, targets.sum())
        for name, score in scores.items():
            ranking = np.lexsort((candidates, -score[section]))[:20]
            hits[name] += targets[ranking].sum()
        ranking_points = []
        for name in ("RRF 3", "all features"):
            order = np.lexsort((candidates, -scores[name][section]))
            points = np.empty(len(order))
            points[order] = 1 / (21 + np.arange(len(order)))
            ranking_points.append(points)
        for weight in (0.25, 0.5):
            combined = weight * ranking_points[0] + (1 - weight) * ranking_points[1]
            ranking = np.lexsort((candidates, -combined))[:20]
            hits[f"all features + RRF 3 ({weight:.0%})"] += targets[ranking].sum()
        pointer = section.stop
    print(f"Candidate ceiling: {coverage}/{total} = {coverage / total:.9f}")
    for name, count in hits.items():
        print(f"{name}: {count}/{total} = {count / total:.9f}")


def fit_reranker(features, labels):
    model = HistGradientBoostingClassifier(
        max_iter=135, max_leaf_nodes=23, learning_rate=0.08, min_samples_leaf=100,
        l2_regularization=2, early_stopping=False, random_state=2026,
    )
    return model.fit(features, labels)


def train_submission_model(features, labels):
    rng = np.random.default_rng(2026)
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    negatives = rng.choice(negatives, size=min(len(negatives), 650000), replace=False)
    rows = np.concatenate((positives, negatives))
    return fit_reranker(features[rows], labels[rows])


def write_submission(args, model):
    with args.sample.open(newline="") as source:
        users = [int(row["user_id"]) for row in csv.DictReader(source)]
    with args.test.open(newline="") as source:
        test = [int(row["user_id"]) for row in csv.DictReader(source)]
    if len(set(users)) != len(users) or sorted(users) != sorted(test):
        raise ValueError("Sample submission and test user lists differ.")

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.submission.with_suffix(".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        rows = candidate_features(args.train, args.train, args.meta, users)
        for index, (user, items, features) in enumerate(rows, 1):
            probabilities = model.predict_proba(features)[:, 1]
            blend = np.zeros(len(items))
            for weight, scores in ((0.75, probabilities), (0.25, features[:, 4])):
                order = np.lexsort((items, -scores))
                blend[order] += weight / (21 + np.arange(len(order)))
            top = items[np.lexsort((items, -blend))[:20]]
            writer.writerow([user, " ".join(map(str, top))])
            if index % 8000 == 0:
                print(f"Ranked {index} submission users.", flush=True)
    temporary.replace(args.submission)
    print(f"Wrote {args.submission}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--meta", type=Path, default=Path("data/item_meta.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("outputs"))
    parser.add_argument("--features", type=Path, help="Reuse an existing local feature cache.")
    parser.add_argument("--submission", type=Path, help="Write full-training Kaggle predictions.")
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    args = parser.parse_args()
    if args.features:
        with np.load(args.features) as cache:
            if args.submission:
                model = train_submission_model(cache["features"], cache["labels"])
            else:
                evaluate(*(cache[name] for name in ("features", "labels", "items", "offsets", "denominators")))
    else:
        data = generate(args.train, args.validation, args.meta)
        if args.submission:
            model = train_submission_model(*data[:2])
            del data
        else:
            evaluate(*data)
    if args.submission:
        write_submission(args, model)


if __name__ == "__main__":
    main()
