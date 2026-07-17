import csv

from model.model import GRAVITY_SPY_CLASSES
from scripts.smoke_test_metadata import validate_metadata


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gravityspy_id", "label", "sample_type"])
        for gs_id, label, split in rows:
            w.writerow([gs_id, label, split])


def test_full_taxonomy_passes(tmp_path):
    csv_path = tmp_path / "good.csv"
    rows = []
    for cls in GRAVITY_SPY_CLASSES:
        for split in ("training",) * 5 + ("validation",) * 2 + ("test",) * 2:
            rows.append((f"{cls[:3]}{len(rows)}", cls, split))
    _write_csv(csv_path, rows)

    report = validate_metadata(str(csv_path))
    assert report.passed
    assert report.missing_classes == set()
    assert len(report.class_counts) == len(GRAVITY_SPY_CLASSES)


def test_missing_class_fails(tmp_path):
    csv_path = tmp_path / "missing_class.csv"
    rows = []
    for cls in GRAVITY_SPY_CLASSES:
        if cls == "Tomte":
            continue
        for split in ("training", "validation", "test"):
            rows.append((f"{cls[:3]}{len(rows)}", cls, split))
    _write_csv(csv_path, rows)

    report = validate_metadata(str(csv_path))
    assert not report.passed
    assert report.missing_classes == {"Tomte"}


def test_unexpected_class_is_reported_but_does_not_fail(tmp_path):
    csv_path = tmp_path / "extra_class.csv"
    rows = []
    for cls in GRAVITY_SPY_CLASSES:
        for split in ("training", "validation", "test"):
            rows.append((f"{cls[:3]}{len(rows)}", cls, split))
    for split in ("training", "validation", "test"):
        rows.append((f"fake{len(rows)}", "Fake_Class", split))
    _write_csv(csv_path, rows)

    report = validate_metadata(str(csv_path))
    assert report.passed  # unexpected classes are informational, not fatal
    assert report.unexpected_classes == {"Fake_Class"}


def test_unmapped_sample_type_is_counted_and_warned(tmp_path):
    csv_path = tmp_path / "unmapped_split.csv"
    rows = [("aaa", "Blip", "training"), ("bbb", "Blip", "engineering_run")]
    _write_csv(csv_path, rows)

    report = validate_metadata(str(csv_path))
    assert report.unmapped_split_rows == 1
    assert any("unrecognized sample_type" in w for w in report.warnings)


def test_missing_gravityspy_id_column_raises(tmp_path):
    csv_path = tmp_path / "wrong_columns.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "class", "split"])
        w.writerow(["aaa", "Blip", "training"])

    import pytest
    with pytest.raises(ValueError, match="gravityspy_id"):
        validate_metadata(str(csv_path))


def test_class_with_zero_val_rows_warns(tmp_path):
    csv_path = tmp_path / "no_val.csv"
    rows = []
    for cls in GRAVITY_SPY_CLASSES:
        splits = ("training", "test") if cls == "Blip" else ("training", "validation", "test")
        for split in splits:
            rows.append((f"{cls[:3]}{len(rows)}", cls, split))
    _write_csv(csv_path, rows)

    report = validate_metadata(str(csv_path))
    assert any("Blip" in w and "validation" in w for w in report.warnings)
