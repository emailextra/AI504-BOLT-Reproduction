"""Verify submission artifacts, embedded notebook source and raw evidence integrity."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def verify(mask_audit=None):
    """Fail on evidence drift, stale notebook code or missing slide requirements."""
    notebook = json.loads((ROOT / "BOLT_Colab_Final.ipynb").read_text(encoding="utf-8"))
    payload = None
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(source, f"notebook cell {index}", "exec")
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "sources" for t in node.targets):
                payload = ast.literal_eval(node.value)
    if not payload:
        raise ValueError("Missing self-contained notebook scripts")
    for name, source in payload.items():
        if source != (ROOT / name).read_text(encoding="utf-8-sig"):
            raise ValueError(f"Embedded notebook source differs: {name}")
    with (ROOT / "results/manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / "results" / row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Raw evidence changed: {row['path']}")
    with zipfile.ZipFile(ROOT / "Slides.pptx") as archive:
        slides = sorted(n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
        notes = [n for n in archive.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        if len(slides) != 13 or len(notes) != 13:
            raise ValueError("Expected thirteen slides with speaker notes")
        for name in slides:
            text = " ".join(ET.fromstring(archive.read(name)).itertext())
            if "Report" not in text and "AI504" not in text and "Project" not in text and "Code" not in text:
                raise ValueError(f"Missing evidence pointer on {name}")
        for number in (2, 4, 5, 9):
            page = ET.fromstring(archive.read(f"ppt/slides/slide{number}.xml"))
            if not page.findall(".//a:tbl", NS):
                raise ValueError(f"Native table missing on slide {number}")
    repo = json.loads((ROOT / "submission-details.json").read_text())["repository_url"]
    if repo != (ROOT / "Repository-link.txt").read_text().strip():
        raise ValueError("Repository link differs across artifacts")
    for name in ("Report.pdf", "Slides.pdf"):
        if not (ROOT / name).read_bytes().startswith(b"%PDF"):
            raise ValueError(f"Invalid PDF signature: {name}")
    if len(list((ROOT / "Papers").glob("*.pdf"))) != 5:
        raise ValueError("Papers folder must contain exactly five PDFs")
    if mask_audit:
        with mask_audit.open(encoding="utf-8", newline="") as handle:
            actual = list(csv.DictReader(handle))
        with (ROOT / "results/table_mask_audit.csv").open(encoding="utf-8", newline="") as handle:
            expected = list(csv.DictReader(handle))
        if actual != expected:
            raise ValueError("Fresh upstream data audit differs from the recorded mask audit")
    print("Package checks passed: source parity, notebook syntax, evidence hashes, slides and paper PDFs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-audit", type=Path)
    args = parser.parse_args()
    verify(args.mask_audit)
