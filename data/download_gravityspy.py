"""
Download the Gravity Spy labeled glitch dataset.

Source: "Updated Gravity Spy Data Set" (Coughlin, 2018), Zenodo record
1476551 -- https://zenodo.org/records/1476551 (DOI 10.5281/zenodo.1476551).
This is the record actually cited by the Gravity Spy ML papers for the
22-class training set; verified against the live Zenodo page (2026-07).

That record ships three files -- this script pulls the two you need for
an image classifier and skips the third by default:
    trainingset_v1d1_metadata.csv   (5.4 MB)  -- always downloaded
    trainingsetv1d1.tar.gz          (5.5 GB)  -- raw PNGs, downloaded by default
    trainingsetv1d1.h5              (3.1 GB)  -- same data as a big HDF5 array;
                                                 redundant with the tar.gz for
                                                 our purposes, use --with-h5 to
                                                 fetch it instead/as well.

IMPORTANT: the PNGs inside trainingsetv1d1.tar.gz still have matplotlib
axis labels/borders baked in -- they are NOT yet a clean spectrogram
image. Run scripts/build_dataset.py after this to crop, resize, and lay
them out as data/processed/{train,val,test}/<class>/*.png (the format
model/dataset.py expects).

This script must be run on a machine with internet access (not the
sandbox this project was originally scaffolded in).

A file this size (5.5GB) is prone to connection drops on ordinary home/
office networks. download_file() below is resumable: if a download is
interrupted, just re-run the script -- it picks up from the last byte
written via an HTTP Range request rather than starting over, and retries
transient failures with backoff. Don't delete a partial file; it's the
resume checkpoint.

Usage:
    python data/download_gravityspy.py --out data/raw
    python data/download_gravityspy.py --out data/raw --with-h5 --skip-tarball
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import requests

ZENODO_API = "https://zenodo.org/api/records/{record_id}"
RECORD_ID = "1476551"  # verified against https://zenodo.org/records/1476551

EXPECTED_FILES = {
    "trainingset_v1d1_metadata.csv",
    "trainingsetv1d1.tar.gz",
    "trainingsetv1d1.h5",
}


def resolve_files(record_id: str):
    resp = requests.get(ZENODO_API.format(record_id=record_id), timeout=30)
    resp.raise_for_status()
    record = resp.json()
    return {
        f["key"]: {"url": f["links"]["self"], "size": f["size"]}
        for f in record.get("files", [])
    }


def download_file(url: str, dest: Path, expected_size: Optional[int] = None,
                   chunk_size: int = 1 << 20, max_retries: int = 5,
                   _sleep=time.sleep):
    """Download `url` to `dest`, resumable across retries.

    If `dest` already has a partial download on disk (e.g. from a prior
    run that got interrupted), resumes via an HTTP Range request instead
    of restarting. If the server doesn't honor Range (returns 200 instead
    of 206), falls back to a clean restart rather than corrupting the file
    with a truncated-then-appended mismatch.

    Verifies the final size against `expected_size` when known (from the
    Zenodo API's own file listing) -- a size mismatch after a "successful"
    stream is exactly the failure mode that produces a corrupt tarball
    that then fails confusingly deep inside a downstream script instead
    of here, where the problem actually is.
    """
    attempt = 0
    while True:
        attempt += 1
        existing_size = dest.stat().st_size if dest.exists() else 0
        if expected_size and existing_size >= expected_size:
            return  # already complete from a prior run

        headers = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if existing_size > 0 and r.status_code == 200:
                    # Server ignored our Range request (doesn't support
                    # resume for this URL) -- must restart clean rather
                    # than append full content onto existing bytes.
                    existing_size = 0
                    mode = "wb"
                r.raise_for_status()

                remaining = int(r.headers.get("content-length", 0))
                total = expected_size or (existing_size + remaining)
                written = existing_size
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        fh.write(chunk)
                        written += len(chunk)
                        if total:
                            pct = 100 * written / total
                            print(f"\r    {dest.name}: {pct:5.1f}% "
                                  f"({written / 1e9:.2f}/{total / 1e9:.2f} GB)",
                                  end="", flush=True)
                print()

            final_size = dest.stat().st_size
            if expected_size and final_size != expected_size:
                raise OSError(
                    f"Downloaded size {final_size} != expected {expected_size} "
                    f"after stream completed without error (connection likely "
                    f"dropped silently)."
                )
            return  # success

        except (requests.exceptions.RequestException, OSError) as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to download {dest.name} after {max_retries} "
                    f"attempts. Last error: {e}\nA partial file is still on "
                    f"disk at {dest} -- re-run this script to resume from "
                    f"where it left off. Do NOT delete the partial file."
                ) from e
            wait = min(2 ** attempt, 30)
            progress = dest.stat().st_size if dest.exists() else 0
            print(f"\n  Download interrupted ({e}); retrying in {wait}s "
                  f"(attempt {attempt}/{max_retries}, {progress / 1e9:.2f} GB "
                  f"written so far) ...")
            _sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--record-id", default=RECORD_ID)
    ap.add_argument("--skip-tarball", action="store_true",
                     help="skip the 5.5GB raw-PNG tarball (rarely useful -- "
                          "you need it or the h5 to train)")
    ap.add_argument("--with-h5", action="store_true",
                     help="also fetch the 3.1GB HDF5 array version")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolving Zenodo record {args.record_id} ...")
    files = resolve_files(args.record_id)
    missing = EXPECTED_FILES - set(files)
    if missing:
        print(f"WARNING: expected files not found in this record: {missing}. "
              f"The record may have changed -- check "
              f"https://zenodo.org/records/{args.record_id} manually.",
              file=sys.stderr)

    to_fetch = ["trainingset_v1d1_metadata.csv"]
    if not args.skip_tarball and "trainingsetv1d1.tar.gz" in files:
        to_fetch.append("trainingsetv1d1.tar.gz")
    if args.with_h5 and "trainingsetv1d1.h5" in files:
        to_fetch.append("trainingsetv1d1.h5")

    for key in to_fetch:
        info = files[key]
        dest = out_dir / key
        if dest.exists() and dest.stat().st_size == info["size"]:
            print(f"  already have {key}, skipping")
            continue
        partial = dest.exists() and dest.stat().st_size > 0
        print(f"  {'resuming' if partial else 'downloading'} {key} "
              f"({info['size'] / 1e9:.2f} GB) ...")
        download_file(info["url"], dest, expected_size=info["size"])

    print(f"\nDone. Raw files are in {out_dir}")
    print("Next: python scripts/build_dataset.py "
          f"--metadata {out_dir}/trainingset_v1d1_metadata.csv "
          f"--tarball {out_dir}/trainingsetv1d1.tar.gz --out data/processed")


if __name__ == "__main__":
    main()
