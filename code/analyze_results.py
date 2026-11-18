"""Verify saved predictions and regenerate every numeric table in the report."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]


def read(path, delimiter=","):
    """Read CSV files with text labels preserved."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write(path, rows):
    """Write stable, human-readable tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_predictions(rows, unknown):
    """Calculate percentage metrics from test predictions with an explicit unknown label."""
    gold = Counter(row["gold"] for row in rows)
    predicted = Counter(row["predicted"] for row in rows)
    correct = Counter(row["gold"] for row in rows if row["gold"] == row["predicted"])
    labels = sorted(set(gold) | set(predicted))
    fs = {label: 200 * correct[label] / (gold[label] + predicted[label]) for label in labels}
    categories = Counter()
    for row in rows:
        if row["gold"] == row["predicted"]:
            category = "correct"
        elif row["gold"] == unknown:
            category = "unknown accepted"
        elif row["predicted"] == unknown:
            category = "known rejected"
        else:
            category = "wrong known class"
        categories[category] += 1
    return {"K-F1": mean(value for label, value in fs.items() if label != unknown),
            "N-F1": fs[unknown], "F1": mean(fs.values()),
            "ACC": 100 * sum(correct.values()) / len(rows)}, categories


def all_tables(output):
    """Check each saved score and calculate cross-fold, paired, runtime, and error tables."""
    published = {(r["method"], r["known_ratio"]): r for r in read(ROOT / "code/published.csv")}
    audit, runtime, errors, paired, comparisons, folds, examples = [], [], [], [], [], [], []
    for record in sorted(output.rglob("run.json")):
        if "work" in record.relative_to(output).parts:
            continue
        directory = record.parent
        config = json.loads(record.read_text())
        command = config["command"]
        epochs = int(command[command.index("--num_train_epochs") + 1])
        name, ratio, fold = directory.name.split("_")
        is_deepunk = name == "deepunk"
        duration = float(config["elapsed_seconds"])
        runtime.append(dict(trial=directory.relative_to(output).as_posix(), seconds=duration,
                            epoch_cap=epochs))
        for row in read(directory / "results.csv"):
            detector = row.get("ood_method", "lof")
            prediction_path = directory / ("predictions.csv" if is_deepunk else f"predictions_{detector}.csv")
            predictions = read(prediction_path)
            if len(predictions) != 3080:
                raise ValueError(f"Unexpected test sample count in {prediction_path}")
            metrics, categories = score_predictions(predictions, "unseen" if is_deepunk else "19")
            scale = 100 if is_deepunk else 1
            for metric, value in metrics.items():
                if abs(float(row[metric]) * scale - value) > .011:
                    raise ValueError(f"Score differs from predictions in {directory}: {metric}")
            audit.append(dict(trial=directory.relative_to(output).as_posix(), detector=detector,
                              test_rows=len(predictions), verified=True))
            context = dict(trial=directory.relative_to(output).as_posix(), detector=detector)
            for category in ("correct", "unknown accepted", "known rejected", "wrong known class"):
                errors.append(dict(context, category=category, count=categories[category]))
            if is_deepunk and directory.parent.name == "full":
                folds.append(dict(known_ratio=ratio, fold=fold, **metrics))
                for index, pred in enumerate(predictions):
                    if pred["gold"] != pred["predicted"] and index in (0, 2, 41) and ratio == "0.25" and fold == "0":
                        examples.append(dict(test_row=index, raw_gold=pred["raw_gold"],
                                             gold=pred["gold"], predicted=pred["predicted"]))
            if "short_5epoch" in directory.parts:
                paired.append(dict(variant=directory.parent.name, detector=detector,
                                   seconds=duration, epoch_cap=epochs, **metrics))
    if len(folds) != 10 or len(paired) != 4:
        raise ValueError("Expected ten DeepUNK fold rows and four paired detector rows")
    for ratio in ("0.25", "0.5"):
        selected = [row for row in folds if row["known_ratio"] == ratio]
        for metric in ("K-F1", "N-F1"):
            values = [row[metric] for row in selected]
            reference = float(published["deepunk", ratio][metric])
            comparisons.append(dict(known_ratio=ratio, metric=metric, published=reference,
                                    mean=round(mean(values), 3), fold_sd=round(stdev(values), 3),
                                    delta=round(mean(values)-reference, 3)))
    table_path = output / "tables"
    for name, rows in [("deepunk_folds", folds), ("deepunk_comparison", comparisons),
                       ("knncon_paired", paired), ("errors", errors), ("runtime", runtime),
                       ("prediction_audit", audit), ("failure_examples", examples)]:
        write(table_path / f"{name}.csv", rows)
    # Existing mask audit was independently generated against the pinned dataset.
    mask = read(output / "table_mask_audit.csv")
    write(table_path / "mask_audit.csv", mask)
    print(f"Verified {len(audit)} score rows against 3,080 predictions each.")
    print("Generated all report tables in", table_path)


def manifest(output):
    """Record SHA-256 for immutable saved evidence and source provenance."""
    rows=[]
    for path in sorted(output.rglob("*")):
        if path.is_file() and "work" not in path.relative_to(output).parts and "tables" not in path.relative_to(output).parts and path.name != "manifest.csv":
            rows.append(dict(path=path.relative_to(output).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(output / "manifest.csv", rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args=parser.parse_args()
    all_tables(args.results)
    manifest(args.results)
