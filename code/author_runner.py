"""Run the released BOLT implementations and audit the BANKING77 label mask."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import subprocess
import sys
from time import monotonic
from urllib.request import urlretrieve
import zipfile

ROOT = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/CNIC-DSL/BOLT.git"
REVISION = "2c935438b5506febcfd23e499e3de81868236b17"
RATIOS = ("0.25", "0.5")
METHODS = {"deepunk": "DeepUnk/experiment.py", "knncon": "KnnCon/run_main.py"}
METRICS = ("K-F1", "N-F1", "ACC", "F1")


def read_rows(path, delimiter=","):
    """Read the author TSV/CSV files without inferring label-mask types."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_rows(path, rows):
    """Write a reproducible table with stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(source):
    """Acquire the immutable author release and the DeepUNK embeddings."""
    if not source.exists():
        subprocess.run(["git", "clone", UPSTREAM, str(source)], check=True)
        subprocess.run(["git", "checkout", "--detach", REVISION], cwd=source, check=True)
    verify_revision(source)
    glove = source / "pretrained_models" / "glove.6B.300d.txt"
    glove.parent.mkdir(exist_ok=True)
    if not glove.exists():
        archive = glove.with_suffix(".zip")
        urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", archive)
        with zipfile.ZipFile(archive) as embeddings:
            temporary = glove.with_suffix(".partial")
            with embeddings.open(glove.name) as original, temporary.open("wb") as target:
                shutil.copyfileobj(original, target)
            temporary.replace(glove)
        archive.unlink()


def verify_revision(source):
    """Fail if the chosen checkout does not identify the pinned release."""
    head = (source / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ")
        ref_path = source / ".git" / ref
        if ref_path.exists():
            head = ref_path.read_text().strip()
        else:
            packed = (source / ".git" / "packed-refs").read_text().splitlines()
            head = next(line.split()[0] for line in packed if line.endswith(" " + ref))
    if head != REVISION:
        raise ValueError(f"Expected BOLT {REVISION}, found {head}")


def audit(source):
    """Verify split alignment and count the effect of the string-to-bool conversion."""
    data = source / "data" / "banking"
    rows = []
    for split in ("train", "dev"):
        original = read_rows(data / "origin_data" / f"{split}.tsv", "\t")
        masked = read_rows(data / "labeled_data/0.1" / f"{split}.tsv", "\t")
        if [row["label"] for row in original] != [row["label"] for row in masked]:
            raise ValueError(f"{split} labels are misaligned")
        if {row["labeled"] for row in masked} != {"True", "False"}:
            raise ValueError(f"Unexpected mask values in {split}")
        for ratio in RATIOS:
            for fold in range(5):
                path = data / f"label/fold5/part{fold}/label_known_{ratio}.list"
                known = set(path.read_text(encoding="utf-8").splitlines())
                candidates = [row for row in masked if row["label"] in known]
                selected = [row for row in candidates if row["labeled"] == "True"]
                if {row["label"] for row in selected} != known:
                    raise ValueError(f"Missing labeled examples for {ratio}, fold {fold}, {split}")
                rows.append(dict(split=split, known_ratio=ratio, fold=fold,
                                 classes=len(known), released_knncon=len(candidates),
                                 corrected_knncon=len(selected)))
    return rows


def replace_once(source, needle, replacement):
    """Anchor instrumentation to the inspected source revision."""
    if source.count(needle) != 1:
        raise ValueError(f"Unexpected upstream source around {needle}")
    return source.replace(needle, replacement)


def stage_method(source, name, corrected, work):
    """Copy a method to scratch and add prediction exports and the optional mask fix."""
    entry = Path(METHODS[name])
    location = work / entry.parent
    shutil.copytree(source / "code/openset/baselines" / entry.parent, location,
                    dirs_exist_ok=True)
    if name == "deepunk":
        path = location / entry.name
        code = path.read_text(encoding="utf-8")
        code = replace_once(code, "    report = classification_report(",
            '    os.makedirs(args.save_results_path, exist_ok=True)\n'
            '    pd.DataFrame({"gold": y_test_mask, "predicted": y_pred, '
            '"raw_gold": df_test["label"]}).to_csv(\n'
            '        os.path.join(args.save_results_path, "predictions.csv"), index=False\n'
            '    )\n\n    report = classification_report(')
    else:
        path = location / "evaluate.py"
        code = path.read_text(encoding="utf-8")
        code = replace_once(code, "                df_to_save = pd.DataFrame([final_results])",
            '                pd.DataFrame({"gold": labels_test, "predicted": predict_label_test}).to_csv(\n'
            '                    os.path.join(self.data_args.save_results_path, '
            'f"predictions_{setting}.csv"), index=False\n'
            '                )\n'
            '                df_to_save = pd.DataFrame([final_results])')
        code = replace_once(code, '"seed","ACC","F1","K-F1","N-F1","args"]',
                            '"seed","ACC","F1","K-F1","N-F1","args","ood_method"]')
        if corrected:
            runner = location / entry.name
            runner_code = runner.read_text(encoding="utf-8")
            for frame in ("df_train", "df_valid"):
                runner_code = replace_once(runner_code, f'{frame}["labeled"].astype(bool)',
                                           f'{frame}["labeled"].eq("True")')
            runner.write_text(runner_code, encoding="utf-8")
    path.write_text(code, encoding="utf-8")
    return location / entry.name


def run_trial(source, output, name, ratio, fold, mode, epochs=50):
    """Run one isolated fold and record command, elapsed time, and console output."""
    corrected = mode == "mask"
    dest = output / mode / f"{name}_{ratio}_{fold}"
    dest.mkdir(parents=True, exist_ok=True)
    work = output / "work" / mode / dest.name
    args = dict(config=source / f"configs/openset/{name}.yaml", dataset="banking",
                known_cls_ratio=ratio, labeled_ratio=0.1, fold_idx=fold, fold_num=5,
                fold_type="fold", seed=2025, num_train_epochs=1 if mode == "quick" else epochs,
                output_dir=work / "checkpoints", save_results_path=dest)
    if name == "knncon":
        args["bert_model"] = "google-bert/bert-base-uncased"
    entry = stage_method(source, name, corrected, work)
    interpreter = os.environ.get("BOLT_TF_PYTHON", sys.executable) if name == "deepunk" else sys.executable
    command = [interpreter, str(entry)]
    for key, value in args.items():
        command.extend((f"--{key}", str(value)))
    env = dict(os.environ, PYTHONHASHSEED="2025", TF_USE_LEGACY_KERAS="1",
               PYTHONUNBUFFERED="1", WANDB_DISABLED="true")
    if name == "deepunk" and env.get("BOLT_TF_LIBS"):
        env["LD_LIBRARY_PATH"] = env["BOLT_TF_LIBS"]
    if (dest / "run.json").exists():
        raise FileExistsError(f"Run already completed in {dest}. Choose another output directory.")
    for path in dest.glob("*.csv"):
        path.unlink()
    start = monotonic()
    print(f"Running {dest.name} ({mode})", flush=True)
    with (dest / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    rows = read_rows(dest / "results.csv")
    expected = 1 if name == "deepunk" else 2
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} result rows in {dest}")
    run_info = dict(command=command, elapsed_seconds=round(monotonic() - start, 2),
                    upstream=REVISION, mode=mode)
    (dest / "run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")


def run_table(output, mode, folds):
    """Aggregate only complete groups and retain both KNNCon detectors."""
    rows = []
    for name in METHODS:
        if mode == "mask" and name == "deepunk":
            continue
        for ratio in RATIOS:
            by_detector = {}
            for fold in folds:
                directory = output / mode / f"{name}_{ratio}_{fold}"
                if not (directory / "run.json").is_file():
                    raise ValueError(f"Incomplete trial {directory}")
                for row in read_rows(directory / "results.csv"):
                    by_detector.setdefault(row.get("ood_method", "lof"), []).append(row)
            for detector, trials in by_detector.items():
                if len(trials) != len(folds):
                    raise ValueError(f"Missing detector output for {name}, {ratio}, {detector}")
                result = dict(method=name, known_ratio=ratio, detector=detector,
                              folds=len(folds))
                for metric in METRICS:
                    values = [float(trial[metric]) for trial in trials]
                    if name == "deepunk":
                        values = [100 * value for value in values]
                    result[metric] = round(mean(values), 3)
                    result[f"{metric}_fold_sd"] = round(stdev(values), 3) if len(values) > 1 else ""
                rows.append(result)
    write_rows(output / f"table_{mode}.csv", rows)


def main():
    """Dispatch preparation, audit, reduced run, full grid, or mask sensitivity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "audit", "quick", "full", "mask", "check"])
    parser.add_argument("--upstream", type=Path, default=ROOT / ".bolt")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    options = parser.parse_args()
    source, output = options.upstream.resolve(), options.output.resolve()
    if options.action == "prepare":
        prepare(source)
        return
    verify_revision(source)
    if options.action in {"audit", "check"}:
        rows = audit(source)
        if options.action == "audit":
            write_rows(output / "table_mask_audit.csv", rows)
        else:
            import tempfile
            with tempfile.TemporaryDirectory() as scratch:
                for method in METHODS:
                    entry = stage_method(source, method, True, Path(scratch))
                    compile(entry.read_text(encoding="utf-8"), str(entry), "exec")
                    if method == "knncon":
                        evaluator = entry.with_name("evaluate.py")
                        compile(evaluator.read_text(encoding="utf-8"), str(evaluator), "exec")
            print(f"Passed split alignment, mask coverage, and patch checks ({len(rows)} split cases)")
        return
    if options.action == "quick":
        run_trial(source, output, "knncon", "0.25", 0, "quick")
        return
    folds = range(5)
    for name in METHODS:
        if options.action == "mask" and name == "deepunk":
            continue
        for ratio in RATIOS:
            for fold in folds:
                dest = output / options.action / f"{name}_{ratio}_{fold}"
                if not (dest / "run.json").exists():
                    run_trial(source, output, name, ratio, fold, options.action)
    run_table(output, options.action, folds)


if __name__ == "__main__":
    main()
