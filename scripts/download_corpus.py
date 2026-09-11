"""
Download the source document corpus (FAA aviation handbooks, public domain).

Usage:
    python scripts/download_corpus.py
    python scripts/download_corpus.py --out data/raw --max-mb 120

The corpus is deliberately NOT committed to git (see .gitignore). Anyone
cloning this repository reproduces it by running this script.

To change domain, replace the CORPUS list below. Nothing else in the project
is aware of the domain.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

USER_AGENT = "rag-assistant-project/1.0 (student coursework; contact via GitHub)"
CHUNK = 1 << 16


@dataclass(frozen=True)
class Document:
    """One source document in the corpus."""

    filename: str
    title: str
    url: str


# ---------------------------------------------------------------------------
# The corpus: FAA handbooks that make up private-pilot ground-school knowledge.
# All are US federal government works and therefore public domain.
# ---------------------------------------------------------------------------
CORPUS: list[Document] = [
    Document(
        filename="phak_pilots_handbook.pdf",
        title="Pilot's Handbook of Aeronautical Knowledge (FAA-H-8083-25B)",
        url="https://www.faa.gov/sites/faa.gov/files/2022-03/pilot_handbook.pdf",
    ),
    Document(
        filename="instrument_flying_handbook.pdf",
        title="Instrument Flying Handbook (FAA-H-8083-15B)",
        url=(
            "https://www.faa.gov/sites/faa.gov/files/regulations_policies/"
            "handbooks_manuals/aviation/FAA-H-8083-15B.pdf"
        ),
    ),
    Document(
        filename="weight_and_balance_handbook.pdf",
        title="Aircraft Weight and Balance Handbook (FAA-H-8083-1B)",
        url=(
            "https://www.faa.gov/sites/faa.gov/files/regulations_policies/"
            "handbooks_manuals/aviation/FAA-H-8083-1.pdf"
        ),
    ),
    Document(
        filename="plane_sense_ga_information.pdf",
        title="Plane Sense: General Aviation Information (FAA-H-8083-19A)",
        url=(
            "https://www.faa.gov/sites/faa.gov/files/regulations_policies/"
            "handbooks_manuals/aviation/faa-h-8083-19A.pdf"
        ),
    ),
    Document(
        filename="remote_pilot_suas_study_guide.pdf",
        title="Remote Pilot - Small UAS Study Guide (FAA-G-8082-22)",
        url=(
            "https://www.faa.gov/sites/faa.gov/files/regulations_policies/"
            "handbooks_manuals/aviation/remote_pilot_study_guide.pdf"
        ),
    ),
]


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def download(doc: Document, out_dir: Path, max_mb: float, force: bool) -> tuple[bool, str]:
    """Download one document. Returns (ok, message)."""
    target = out_dir / doc.filename

    if target.exists() and not force:
        return True, f"already present ({human_mb(target.stat().st_size)}) - skipped"

    tmp = target.with_suffix(target.suffix + ".part")
    limit = int(max_mb * 1024 * 1024)

    try:
        with requests.get(
            doc.url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}
        ) as response:
            response.raise_for_status()

            declared = response.headers.get("Content-Length")
            if declared and int(declared) > limit:
                return False, f"too large ({human_mb(int(declared))} > {max_mb} MB limit)"

            written = 0
            with tmp.open("wb") as handle:
                for block in response.iter_content(CHUNK):
                    written += len(block)
                    if written > limit:
                        tmp.unlink(missing_ok=True)
                        return False, f"exceeded {max_mb} MB limit mid-download"
                    handle.write(block)

        # A wrong URL usually returns an HTML error page with a 200 status.
        with tmp.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                tmp.unlink(missing_ok=True)
                return False, "server did not return a PDF (check the URL)"

        tmp.replace(target)
        return True, f"downloaded {human_mb(written)}"

    except requests.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        return False, f"HTTP {exc.response.status_code}"
    except requests.RequestException as exc:
        tmp.unlink(missing_ok=True)
        return False, f"network error: {exc.__class__.__name__}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the RAG source corpus.")
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument("--max-mb", type=float, default=120.0, help="per-file size cap")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(CORPUS)} documents into {out_dir.resolve()}\n")

    failures: list[tuple[Document, str]] = []
    for doc in CORPUS:
        print(f"  {doc.filename:<40}", end="", flush=True)
        ok, message = download(doc, out_dir, args.max_mb, args.force)
        print(message)
        if not ok:
            failures.append((doc, message))

    print()
    got = len(CORPUS) - len(failures)
    print(f"{got}/{len(CORPUS)} documents available in {out_dir}")

    if failures:
        print("\nThe following could not be downloaded automatically.")
        print("Download them manually into the same folder, then re-run the notebook:\n")
        for doc, message in failures:
            print(f"  - {doc.title}\n    {doc.url}\n    reason: {message}\n")
        # The pipeline works with a partial corpus, so this is a warning, not a failure.
        return 0 if got else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
