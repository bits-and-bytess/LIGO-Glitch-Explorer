import csv
import io
import tarfile

from PIL import Image

from scripts.build_dataset import CROP_BOX, crop_and_resize, load_metadata


def _make_metadata_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gravityspy_id", "label", "sample_type"])
        for gs_id, label, split in rows:
            w.writerow([gs_id, label, split])


def test_load_metadata_maps_sample_type_to_split(tmp_path):
    csv_path = tmp_path / "meta.csv"
    _make_metadata_csv(csv_path, [
        ("aaaa", "Blip", "training"),
        ("bbbb", "Blip", "validation"),
        ("cccc", "Whistle", "test"),
    ])
    rows = load_metadata(str(csv_path), duration="1.0")
    splits = {gs_id: split for gs_id, _, split in rows}
    assert splits == {"aaaa": "train", "bbbb": "val", "cccc": "test"}


def test_load_metadata_skips_rows_missing_required_fields(tmp_path):
    csv_path = tmp_path / "meta.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gravityspy_id", "label", "sample_type"])
        w.writerow(["aaaa", "Blip", "training"])
        w.writerow(["", "Blip", "training"])  # missing id
        w.writerow(["bbbb", "", "training"])  # missing label
        w.writerow(["cccc", "Blip", "unknown_split"])  # unmapped split
    rows = load_metadata(str(csv_path), duration="1.0")
    assert [r[0] for r in rows] == ["aaaa"]


def test_crop_and_resize_strips_border_and_resizes():
    # Build an image with a distinct 'border' color everywhere and a
    # distinct 'content' color inside exactly the documented crop box,
    # matching the real dataset's documented axis-cropping convention.
    img = Image.new("RGB", (700, 600), color=(255, 0, 0))
    left, upper, right, lower = CROP_BOX
    for x in range(left, right):
        for y in range(upper, lower):
            img.putpixel((x, y), (0, 255, 0))

    result = crop_and_resize(img)
    assert result.size == (224, 224)
    import numpy as np
    arr = np.array(result)
    mean_color = arr.reshape(-1, 3).mean(axis=0)
    # Should be essentially pure green -- if the crop box were wrong,
    # some red border would leak into the resized output.
    assert mean_color[1] > 250
    assert mean_color[0] < 5


def test_crop_and_resize_skips_crop_on_undersized_image():
    # An image smaller than the crop box should be resized directly,
    # not crash or produce a zero-size crop.
    img = Image.new("RGB", (50, 50), color=(10, 20, 30))
    result = crop_and_resize(img)
    assert result.size == (224, 224)


def _build_test_tarball(tar_path, samples):
    """samples: list of (gravityspy_id, label, split, duration)"""
    with tarfile.open(tar_path, "w:gz") as tar:
        for gs_id, label, split, duration in samples:
            img = Image.new("RGB", (700, 600), color=(0, 255, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            data = buf.read()
            info = tarfile.TarInfo(name=f"{label}/{split}/H1_{gs_id}_spectrogram_{duration}.png")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_end_to_end_build_writes_only_requested_duration(tmp_path):
    import subprocess
    import sys

    csv_path = tmp_path / "meta.csv"
    tar_path = tmp_path / "data.tar.gz"
    out_dir = tmp_path / "processed"

    _make_metadata_csv(csv_path, [("aaaa", "Blip", "training"), ("bbbb", "Whistle", "test")])
    _build_test_tarball(tar_path, [
        ("aaaa", "Blip", "training", "0.5"),
        ("aaaa", "Blip", "training", "1.0"),
        ("bbbb", "Whistle", "test", "1.0"),
        ("bbbb", "Whistle", "test", "4.0"),
    ])

    # Run from repo root so `-m scripts.build_dataset` resolves correctly
    import pathlib
    repo_root = pathlib.Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_dataset",
         "--metadata", str(csv_path), "--tarball", str(tar_path),
         "--out", str(out_dir), "--duration", "1.0"],
        capture_output=True, text=True, cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr

    assert (out_dir / "train" / "Blip" / "aaaa.png").exists()
    assert (out_dir / "test" / "Whistle" / "bbbb.png").exists()
    # the 0.5s and 4.0s duration files should NOT have been written
    assert not (out_dir / "train" / "Blip" / "aaaa_0.5.png").exists()
