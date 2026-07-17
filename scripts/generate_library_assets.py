"""
Pre-generate the Glitch Library's static JSON + example images so the
library pages never have to call the model at request time.

For each class: pick a few representative validation-set examples, run
inference + GradCAM once, save to backend/static/library/, and write a
classes.json manifest matching backend/schemas.GlitchClassSummary.

Physical-origin descriptions and known-similar-classes below are a
starting point based on published Gravity Spy characterizations -- refine
these with your own reading for the public Methodology / Library pages.
"""
import json
import random
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

from model.dataset import build_transforms
from model.gradcam import GradCAMExplainer
from model.model import load_model

OUT_DIR = Path(__file__).parent.parent / "backend" / "static" / "library"
WEIGHTS_PATH = Path(__file__).parent.parent / "model" / "weights" / "efficientnet_gravityspy.pt"

# Starting-point descriptions. These are intentionally brief placeholders;
# expand with citations to the Gravity Spy paper / Zoo descriptions before
# publishing.
CLASS_INFO = {
    "Blip":               dict(freq=(30, 2000), dur=(0.05, 0.3), origin="Unknown; broadband transient, likely multiple instrumental causes"),
    "Whistle":            dict(freq=(200, 2000), dur=(0.1, 1.0), origin="RF beat notes between auxiliary oscillators"),
    "Koi_Fish":           dict(freq=(20, 500),   dur=(0.2, 1.0), origin="Related to scattered light arches"),
    "Scattered_Light":    dict(freq=(10, 100),   dur=(1.0, 4.0), origin="Light scattering off moving optical surfaces, often ground-motion driven"),
    "Violin_Mode":        dict(freq=(500, 520),  dur=(0.5, 4.0), origin="Resonant modes of suspension fibers"),
    "Power_Line":         dict(freq=(59, 61),     dur=(1.0, 4.0), origin="60 Hz mains power harmonics"),
    "Low_Frequency_Lines": dict(freq=(10, 100),  dur=(1.0, 4.0), origin="Various low-frequency instrumental line noise"),
    "1080Lines":          dict(freq=(1075, 1085), dur=(1.0, 4.0), origin="Fixed-frequency instrumental line near 1080 Hz"),
    "1400Ripples":        dict(freq=(1380, 1420), dur=(0.5, 2.0), origin="Modulated line near 1400 Hz"),
    "Air_Compressor":     dict(freq=(30, 90),    dur=(0.5, 2.0), origin="Facility air-handling equipment"),
    "Chirp":              dict(freq=(20, 500),   dur=(0.2, 2.0), origin="Frequency-sweeping transient; superficially CBC-like"),
    "Extremely_Loud":     dict(freq=(10, 2000),  dur=(0.1, 4.0), origin="High-amplitude broadband transient, often saturation-related"),
    "Helix":              dict(freq=(20, 500),   dur=(0.5, 2.0), origin="Curved time-frequency track, cause not firmly established"),
    "Light_Modulation":    dict(freq=(1, 50),     dur=(1.0, 4.0), origin="Periodic amplitude modulation of scattered/stray light"),
    "Low_Frequency_Burst": dict(freq=(10, 100),  dur=(0.2, 2.0), origin="Broadband low-frequency transient, often ground-motion related"),
    "No_Glitch":          dict(freq=(10, 2048),  dur=(1.0, 4.0), origin="No significant glitch present"),
    "None_of_the_Above":  dict(freq=(10, 2048),  dur=(0.5, 4.0), origin="Doesn't match a known morphology -- candidate for new-class review"),
    "Paired_Doves":       dict(freq=(50, 400),   dur=(0.2, 1.0), origin="Two similar overlapping short transients"),
    "Repeating_Blips":    dict(freq=(30, 500),   dur=(1.0, 4.0), origin="Multiple Blip-like transients recurring in one segment"),
    "Scratchy":           dict(freq=(10, 200),   dur=(1.0, 4.0), origin="Broadband low-frequency noise, textured appearance"),
    "Tomte":              dict(freq=(10, 100),   dur=(0.2, 1.0), origin="Short, low-frequency triangular-shaped transient"),
    "Wandering_Line":     dict(freq=(10, 2000),  dur=(1.0, 4.0), origin="Slowly frequency-drifting instrumental line"),
}

SIMILAR = {
    "Blip": ["Tomte", "Koi_Fish"],
    "Koi_Fish": ["Blip", "Scattered_Light"],
    "Whistle": ["Wandering_Line"],
    "1080Lines": ["1400Ripples", "Violin_Mode"],
    "Power_Line": ["Low_Frequency_Lines"],
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(str(WEIGHTS_PATH), device=device)
    gradcam = GradCAMExplainer(model)
    val_ds = ImageFolder(
        Path("data/processed/val"), transform=build_transforms(train=False)
    )

    # group sample indices by class
    by_class: dict[str, list[int]] = {c: [] for c in val_ds.classes}
    for idx, (_, label) in enumerate(val_ds.samples):
        by_class[val_ds.classes[label]].append(idx)

    manifest = []
    for cls_name in val_ds.classes:
        indices = by_class.get(cls_name, [])
        if not indices:
            continue
        chosen = random.sample(indices, k=min(3, len(indices)))

        cls_dir = OUT_DIR / cls_name
        cls_dir.mkdir(exist_ok=True)
        example_paths = []
        for i, idx in enumerate(chosen):
            image_tensor, label = val_ds[idx]
            raw_path, _ = val_ds.samples[idx]
            from PIL import Image
            raw_img = Image.open(raw_path).convert("RGB").resize((224, 224))

            gradcam_bytes = gradcam.overlay_png_bytes(
                image_tensor.unsqueeze(0).to(device), __import__("numpy").array(raw_img), class_idx=label
            )
            (cls_dir / f"example_{i}.png").write_bytes(gradcam_bytes)
            raw_img.save(cls_dir / f"spectrogram_{i}.png")
            example_paths.append({
                "spectrogram_url": f"/static/library/{cls_name}/spectrogram_{i}.png",
                "gradcam_url": f"/static/library/{cls_name}/example_{i}.png",
            })

        info = CLASS_INFO.get(cls_name, dict(freq=(10, 2048), dur=(0.5, 4.0), origin=None))
        manifest.append({
            "name": cls_name,
            "description": f"Representative {cls_name.replace('_', ' ')} glitch examples "
                            f"with GradCAM overlays showing the time-frequency features "
                            f"the model keys on.",
            "physical_origin": info["origin"],
            "frequency_range_hz": info["freq"],
            "typical_duration_s": info["dur"],
            "example_spectrogram_url": example_paths[0]["spectrogram_url"],
            "example_gradcam_url": example_paths[0]["gradcam_url"],
            "all_examples": example_paths,
            "similar_classes": SIMILAR.get(cls_name, []),
        })

    (OUT_DIR / "classes.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} class entries to {OUT_DIR / 'classes.json'}")


if __name__ == "__main__":
    main()
