"""
Convert the raw Gravity Spy download into the ImageFolder layout
model/dataset.py expects:

    data/processed/{train,val,test}/<ClassName>/<gravityspy_id>.png

Two things the raw data needs that this script handles:

1. **Duration selection.** Each Gravity Spy sample ships as 4 images
   (0.5s/1.0s/2.0s/4.0s duration views), named like
   H1_<gravityspy_id>_spectrogram_<duration>.png inside
   trainingsetv1d1.tar.gz, under /<label>/<sample_type>/. We pick one
   duration per --duration (default 1.0s, matching the spec's default)
   so each sample becomes exactly one training image.

2. **Axis cropping.** The raw PNGs still have matplotlib axes/labels
   baked in. Per the dataset's own documentation (Zenodo record 1476551),
   crop to image[66:532, 105:671, :3] before resizing to the CNN's
   expected input. This script applies that crop and resizes to
   IMAGE_SIZE (224x224) so the output is ready for model/dataset.py
   with no further processing.

Usage:
    python scripts/build_dataset.py \
        --metadata data/raw/trainingset_v1d1_metadata.csv \
        --tarball data/raw/trainingsetv1d1.tar.gz \
        --out data/processed \
        --duration 1.0
"""
import argparse
import csv
import tarfile
from pathlib import Path

from PIL import Image

CROP_BOX = (105, 66, 671, 532)  # PIL crop() takes (left, upper, right, lower)
IMAGE_SIZE = (224, 224)

# Gravity Spy's sample_type values -> our train/val/test folder names.
SPLIT_MAP = {"train": "train", "training": "train",
             "validation": "val", "val": "val",
             "test": "test", "testing": "test"}


def load_metadata(metadata_path: str, duration: str):
    """Returns list of (gravityspy_id, label, split) for every row."""
    rows = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gs_id = row.get("gravityspy_id")
            label = row.get("label")
            sample_type = (row.get("sample_type") or "").lower()
            split = SPLIT_MAP.get(sample_type)
            if not gs_id or not label or not split:
                continue
            rows.append((gs_id, label, split))
    return rows


def crop_and_resize(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    # Only apply the documented crop if the image is actually big enough
    # for it -- guards against a future dataset revision changing raw
    # image dimensions and silently producing garbage crops.
    if w >= CROP_BOX[2] and h >= CROP_BOX[3]:
        img = img.crop(CROP_BOX)
    return img.resize(IMAGE_SIZE, Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", default="1.0", choices=["0.5", "1.0", "2.0", "4.0"])
    ap.add_argument("--limit", type=int, default=None,
                     help="only process the first N metadata rows (for a quick smoke test)")
    args = ap.parse_args()

    out_root = Path(args.out)
    rows = load_metadata(args.metadata, args.duration)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} samples from metadata (duration={args.duration}s)")

    # Build a lookup of gravityspy_id -> (label, split) so we can match
    # tarball members without re-scanning the CSV per file.
    by_id = {gs_id: (label, split) for gs_id, label, split in rows}

    written, skipped_no_match, skipped_wrong_duration, already_existed = 0, 0, 0, 0

    with tarfile.open(args.tarball, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            fname = Path(member.name).name  # e.g. H1_<id>_spectrogram_1.0.png
            if not fname.endswith(f"_{args.duration}.png"):
                skipped_wrong_duration += 1
                continue

            # gravityspy_id is the middle token: <ifo>_<id>_spectrogram_<dur>.png
            parts = fname.replace(".png", "").split("_")
            if len(parts) < 3:
                skipped_no_match += 1
                continue
            gs_id = parts[1]

            match = by_id.get(gs_id)
            if not match:
                skipped_no_match += 1
                continue
            label, split = match

            dest_dir = out_root / split / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{gs_id}.png"
            if dest_path.exists():
                already_existed += 1
                continue

            fh = tar.extractfile(member)
            if fh is None:
                skipped_no_match += 1
                continue
            img = Image.open(fh)
            img = crop_and_resize(img)
            img.save(dest_path)
            written += 1

            if written % 500 == 0:
                print(f"  ... {written} images written")

    print(f"\nDone. Wrote {written} new images to {out_root}")
    print(f"Already present from a prior run: {already_existed}")
    print(f"Total images now in {out_root}: {written + already_existed}")
    print(f"Skipped (wrong duration): {skipped_wrong_duration}")
    print(f"Skipped (no metadata match): {skipped_no_match}")
    print("\nSanity-check the class folder counts before training:")
    print(f"  find {out_root} -mindepth 2 -maxdepth 2 -type d | sort")


if __name__ == "__main__":
    main()
