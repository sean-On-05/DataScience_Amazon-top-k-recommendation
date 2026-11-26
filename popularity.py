import argparse
import csv
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from random import Random

parser = argparse.ArgumentParser(description="Evaluate a popularity recommender.")
parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
parser.add_argument("--output", type=Path, default=Path("outputs"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
train_path = args.train
histories = defaultdict(set)
with train_path.open() as source:
    for row in csv.DictReader(source):
        histories[int(row["user_id"])].add(int(row["item_id"]))

rng = Random(42)
fit = {}
truth = {}
for user, items in sorted(histories.items()):
    holdout_size = min(len(items) - 1, max(1, (len(items) + 5) // 10))
    held = set(rng.sample(sorted(items), holdout_size))
    fit[user] = items - held
    if held:
        truth[user] = held

popularity = Counter(item for items in fit.values() for item in items)
ranking = sorted(popularity, key=lambda item: (-popularity[item], item))

hits = 0
denominator = 0
with (args.output / "validation_recommendations.csv").open("w") as output:
    writer = csv.writer(output)
    writer.writerow(["user_id", "item_id"])
    for user, seen in fit.items():
        recommended = list(islice((item for item in ranking if item not in seen), 20))
        assert len(recommended) == len(set(recommended)) == 20
        writer.writerow([user, " ".join(map(str, recommended))])
        if user in truth:
            hits += len(truth[user].intersection(recommended))
            denominator += min(len(truth[user]), 20)
for name, data in (("validation_train", fit), ("validation_truth", truth)):
    with (args.output / f"{name}.csv").open("w") as output:
        writer = csv.writer(output)
        writer.writerow(["user_id", "item_id"])
        writer.writerows((user, item) for user, items in data.items() for item in sorted(items))
print("users", len(fit), len(truth))
print("fit and held", sum(popularity.values()), sum(map(len, truth.values())))
print("hits denominator recall", hits, denominator, hits / denominator)
print("top", [(item, popularity[item]) for item in ranking[:5]])
