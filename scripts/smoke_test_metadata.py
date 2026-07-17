"""
Fast pre-flight check before pulling the full 5.5GB Gravity Spy tarball.

Downloads only trainingset_v1d1_metadata.csv (5.4MB) and validates it
against what the rest of this project assumes:
    - every class in model.GRAVITY_SPY_CLASSES actually appears
    - no unexpected classes show up that we haven't accounted for
    - train/val/test split sizes look sane (not empty, not wildly skewed)

Run this FIRST. If it passes, the full tarball pull + build_dataset.py
should go smoothly. If it flags mismatches, fix those before spending
time on the 5.5GB download.

Usage:
    python scripts/smoke_test_metadata.py --out data/raw
    python scripts/smoke_test_metadata.py --csv data/raw/trainingset_v1d1_metadata.csv  # skip download, re-check existing file
"""
import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.download_gravityspy import RECORD_ID, download_file, resolve_files  # noqa: E402
from model.model import GRAVITY_SPY_CLASSES  # noqa: E402

METADATA_FILENAME = "trainingset_v1d1_metadata.csv"
SPLIT_MAP = {"train": "train", "training": "train",
             "validation": "val", "val": "val",
             "test": "test", "testing": "test"}


@dataclass
class SmokeTestReport:
    total_rows: int
    class_counts: Counter
    split_counts: dict  # {class: {split: count}}
    missing_classes: set    # in GRAVITY_SPY_CLASSES but not in the data
    unexpected_classes: set  # in the data but not in GRAVITY_SPY_CLASSES
    unmapped_split_rows: int
    warnings: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # Unexpected classes are just informational (dataset may have grown);
        # missing classes are the real problem, since inference would then
        # rely on a class the model was never actually trained to recognize.
        return len(self.missing_classes) == 0 and self.total_rows > 0


def validate_metadata(csv_path: str) -> SmokeTestReport:
    """Pure logic, no network -- takes a local CSV path and reports on it.
    Split out from download() specifically so this is unit-testable without
    hitting Zenodo.
    """
    class_counts = Counter()
    split_counts: dict = {}
    total_rows = 0
    unmapped_split_rows = 0
    warnings = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "gravityspy_id" not in reader.fieldnames:
            raise ValueError(
                f"'{csv_path}' doesn't look like a Gravity Spy metadata CSV "
                f"(no gravityspy_id column). Columns found: {reader.fieldnames}"
            )
        for row in reader:
            total_rows += 1
            label = row.get("label")
            sample_type = (row.get("sample_type") or "").lower()
            split = SPLIT_MAP.get(sample_type)
            if not label:
                continue
            class_counts[label] += 1
            if split:
                split_counts.setdefault(label, Counter())[split] += 1
            else:
                unmapped_split_rows += 1

    found_classes = set(class_counts)
    expected_classes = set(GRAVITY_SPY_CLASSES)
    missing = expected_classes - found_classes
    unexpected = found_classes - expected_classes

    if unmapped_split_rows > 0:
        warnings.append(
            f"{unmapped_split_rows} rows had an unrecognized sample_type "
            f"value (not train/validation/test) and were skipped."
        )
    for cls, counts in split_counts.items():
        if counts.get("train", 0) == 0:
            warnings.append(f"Class '{cls}' has zero training rows.")
        if counts.get("val", 0) == 0:
            warnings.append(f"Class '{cls}' has zero validation rows.")

    return SmokeTestReport(
        total_rows=total_rows,
        class_counts=class_counts,
        split_counts=split_counts,
        missing_classes=missing,
        unexpected_classes=unexpected,
        unmapped_split_rows=unmapped_split_rows,
        warnings=warnings,
    )


def print_report(report: SmokeTestReport):
    print(f"\nTotal rows: {report.total_rows}")
    print(f"Classes found: {len(report.class_counts)} / expected {len(GRAVITY_SPY_CLASSES)}")

    print("\nPer-class row counts (train / val / test):")
    for cls in sorted(GRAVITY_SPY_CLASSES):
        counts = report.split_counts.get(cls, {})
        if cls in report.missing_classes:
            print(f"  {cls:<22} MISSING FROM DATASET")
        else:
            print(f"  {cls:<22} {counts.get('train', 0):>5} / "
                  f"{counts.get('val', 0):>5} / {counts.get('test', 0):>5}")

    if report.unexpected_classes:
        print(f"\nClasses present in data but not in model.GRAVITY_SPY_CLASSES: "
              f"{sorted(report.unexpected_classes)}")
        print("  -> either the dataset has grown since this list was last "
              "checked, or these are near-duplicate label variants. Update "
              "model/model.py's GRAVITY_SPY_CLASSES if legitimate.")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings:
            print(f"  - {w}")

    print(f"\n{'PASSED' if report.passed else 'FAILED'}: ", end="")
    if report.passed:
        print("all expected classes present, safe to proceed with the full pull.")
    else:
        print(f"missing classes: {sorted(report.missing_classes)}")
        print("  -> do NOT proceed to the full 5.5GB pull until this is resolved "
              "(training would silently produce a model that can never predict "
              "the missing class).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="directory to download the CSV into")
    ap.add_argument("--csv", default=None,
                     help="skip download, validate an existing local CSV instead")
    ap.add_argument("--record-id", default=RECORD_ID)
    args = ap.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"No such file: {csv_path}", file=sys.stderr)
            sys.exit(2)
    else:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / METADATA_FILENAME
        if csv_path.exists():
            print(f"{csv_path} already exists, re-validating without re-downloading.")
        else:
            print(f"Resolving Zenodo record {args.record_id} ...")
            files = resolve_files(args.record_id)
            if METADATA_FILENAME not in files:
                print(f"'{METADATA_FILENAME}' not found in record {args.record_id}. "
                      f"Files present: {list(files)}", file=sys.stderr)
                sys.exit(1)
            print(f"Downloading {METADATA_FILENAME} "
                  f"({files[METADATA_FILENAME]['size'] / 1e6:.1f} MB) ...")
            download_file(files[METADATA_FILENAME]["url"], csv_path)

    report = validate_metadata(str(csv_path))
    print_report(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
