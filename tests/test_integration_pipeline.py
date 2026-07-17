"""
Integration test for the full offline pipeline: train a tiny model on
synthetic data, calibrate the OOD threshold against it, then generate
library assets from it.

Deliberately invoked via subprocess (as `python model/train.py ...`, the
exact way a person copy-pasting the README would run it) rather than
imported and called directly -- that's what originally caught the
sys.path bug where `python model/train.py` failed but `python -m
model.train` didn't. Importing the module directly would have missed it.

This test is slower than the rest of the suite (~15-30s) since it does
real (tiny) training. That's an intentional tradeoff: it's the only test
that exercises the full train -> calibrate -> library-assets chain
end-to-end, which is exactly where cross-script integration bugs
(hardcoded paths, import resolution, checkpoint format mismatches) show
up that no individual unit test would catch.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from model.model import GRAVITY_SPY_CLASSES

REPO_ROOT = Path(__file__).parent.parent


def _build_tiny_synthetic_dataset(root: Path):
    """All 22 classes (required by dataset.py's sanity check), minimal
    images per split, small source resolution (the training transform
    resizes to 224x224 regardless, so this is fast to generate/write).
    """
    rng = np.random.default_rng(0)
    for i, cls in enumerate(GRAVITY_SPY_CLASSES):
        base_color = np.array([(i * 37) % 255, (i * 97) % 255, (i * 151) % 255])
        for split, n in [("train", 3), ("val", 1), ("test", 1)]:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for j in range(n):
                noise = rng.integers(-20, 20, size=(32, 32, 3))
                img = np.clip(base_color[None, None, :] + noise, 0, 255).astype("uint8")
                Image.fromarray(img).save(d / f"{j}.png")


def _run(cmd):
    result = subprocess.run(
        [sys.executable] + cmd, capture_output=True, text=True, cwd=str(REPO_ROOT)
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}"
        f"\n--- stderr ---\n{result.stderr}"
    )
    return result


def test_train_calibrate_and_generate_library_assets_end_to_end(tmp_path):
    data_root = tmp_path / "processed"
    _build_tiny_synthetic_dataset(data_root)

    weights_path = tmp_path / "weights" / "model.pt"
    ood_path = tmp_path / "weights" / "ood_threshold.json"
    library_out = tmp_path / "library"

    # --- Step 1: train (as `python model/train.py`, not `-m model.train`,
    # to specifically guard against the sys.path regression this caught) ---
    _run([
        "model/train.py",
        "--data", str(data_root),
        "--epochs", "1",
        "--batch-size", "4",
        "--num-workers", "0",
        "--no-pretrained",
        "--out", str(weights_path),
    ])
    assert weights_path.exists()
    history = json.loads(weights_path.with_suffix(".history.json").read_text())
    assert len(history["history"]) == 1
    assert history["classes"] == sorted(GRAVITY_SPY_CLASSES)

    # --- Step 2: calibrate OOD threshold against the freshly trained model ---
    _run([
        "scripts/calibrate_ood.py",
        "--data", str(data_root),
        "--weights", str(weights_path),
        "--out", str(ood_path),
        "--num-workers", "0",
    ])
    assert ood_path.exists()
    ood_data = json.loads(ood_path.read_text())
    assert "threshold" in ood_data

    # --- Step 3: generate library assets from the same model ---
    _run([
        "scripts/generate_library_assets.py",
        "--data", str(data_root),
        "--weights", str(weights_path),
        "--out", str(library_out),
    ])
    manifest = json.loads((library_out / "classes.json").read_text())
    assert len(manifest) == len(GRAVITY_SPY_CLASSES)
    for entry in manifest:
        assert (library_out / entry["name"] / "example_0.png").exists()
        assert (library_out / entry["name"] / "spectrogram_0.png").exists()
