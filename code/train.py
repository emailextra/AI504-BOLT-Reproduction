"""Run the completed study protocol using the pinned author implementation."""
import argparse
import os
from pathlib import Path
import subprocess
from urllib.request import urlretrieve
from zipfile import ZipFile
import shutil

from author_runner import REVISION, UPSTREAM, run_trial, verify_revision

ROOT = Path(__file__).resolve().parents[1]


def acquire(source):
    """Fetch the recorded author revision without installing model dependencies."""
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True)
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=source,
                          capture_output=True, text=True)
    if head.returncode != 0:
        subprocess.run(["git", "fetch", "--depth", "1", UPSTREAM, REVISION],
                       cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source,
                       check=True)
    verify_revision(source)


def embeddings(source):
    """Download and extract only the required GloVe embedding file."""
    target = source / "pretrained_models/glove.6B.300d.txt"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return
    archive = target.parent / "glove.6B.zip"
    urlretrieve("https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip", archive)
    partial = target.with_suffix(".partial")
    with ZipFile(archive) as bundle:
        with bundle.open(target.name) as original, partial.open("wb") as output:
            shutil.copyfileobj(original, output)
    partial.replace(target)
    archive.unlink()


def main():
    """Execute one report experiment, retaining previous successful trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["prepare", "deepunk", "quick", "paired"])
    parser.add_argument("--upstream", type=Path, default=ROOT / ".bolt")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    source, output = args.upstream.resolve(), args.output.resolve()
    acquire(source)
    os.environ.setdefault("MPLBACKEND", "Agg")
    if args.experiment == "prepare":
        return
    if args.experiment == "deepunk":
        needed = [(ratio, fold) for ratio in ("0.25", "0.5") for fold in range(5)
                  if not (output / "full" / f"deepunk_{ratio}_{fold}" / "run.json").exists()]
        if needed:
            embeddings(source)
        for ratio in ("0.25", "0.5"):
            for fold in range(5):
                directory = output / "full" / f"deepunk_{ratio}_{fold}"
                if not (directory / "run.json").exists():
                    run_trial(source, output, "deepunk", ratio, fold, "full", epochs=50)
    elif args.experiment == "quick":
        directory = output / "quick/knncon_0.25_0"
        if not (directory / "run.json").exists():
            run_trial(source, output, "knncon", "0.25", 0, "quick", epochs=1)
    else:
        for mode in ("full", "mask"):
            directory = output / "short_5epoch" / mode / "knncon_0.25_0"
            if not (directory / "run.json").exists():
                run_trial(source, output / "short_5epoch", "knncon", "0.25", 0, mode, epochs=5)


if __name__ == "__main__":
    main()
