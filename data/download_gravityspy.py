"""
Download the Gravity Spy labeled glitch dataset.

Gravity Spy metadata + images are hosted on Zenodo. The canonical dataset
record is "Gravity Spy Training Set" (Bahaadini et al.), which ships as a
set of per-class image folders plus a metadata CSV mapping each sample to:
    - gravityspy_id
    - ml_label (one of ~22 classes)
    - ifo (H1 / L1)
    - event_time (GPS)
    - url_1 / url_2 / url_3 / url_4  (different duration views: 0.5s/1s/2s/4s)

This script must be run on a machine with internet access (not this
sandbox). It resolves the current Zenodo record ID via the Gravity Spy
project page if not supplied, then downloads and unpacks the archive.

Usage:
    python download_gravityspy.py --out data/raw
    python download_gravityspy.py --out data/raw --record-id 1476551
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

ZENODO_API = "https://zenodo.org/api/records/{record_id}"

# Known-good Gravity Spy Zenodo record as of last manual check.
# Pass --record-id to override if this becomes stale.
DEFAULT_RECORD_ID = "1476551"


def resolve_files(record_id: str):
    resp = requests.get(ZENODO_API.format(record_id=record_id), timeout=30)
    resp.raise_for_status()
    record = resp.json()
    return [
        {"key": f["key"], "url": f["links"]["self"], "size": f["size"]}
        for f in record.get("files", [])
    ]


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fh.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--record-id", default=DEFAULT_RECORD_ID)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolving Zenodo record {args.record_id} ...")
    files = resolve_files(args.record_id)
    if not files:
        print("No files found for this record ID. Check the record ID is "
              "current at https://zenodo.org (search 'Gravity Spy').",
              file=sys.stderr)
        sys.exit(1)

    for f in files:
        dest = out_dir / f["key"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"  already have {f['key']}, skipping")
            continue
        print(f"  downloading {f['key']} ({f['size'] / 1e6:.1f} MB) ...")
        download_file(f["url"], dest)

        if not args.skip_extract and dest.suffix == ".zip":
            print(f"  extracting {dest.name} ...")
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(out_dir)

    print(f"Done. Raw Gravity Spy data is in {out_dir}")
    print("Expect a metadata CSV (trainingset_v1d1_metadata.csv or similar) "
          "plus per-class image directories. Point "
          "preprocessing/qtransform.py --metadata at that CSV.")


if __name__ == "__main__":
    main()
