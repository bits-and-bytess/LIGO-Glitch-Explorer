"""
Scan real O4 GWOSC data over a GPS time range, run the trained classifier
+ OOD scorer on sliding windows, and save the highest-OOD-score signals as
the curated Anomaly Gallery (weeks 6-7 deliverable).

Must run on a machine with internet access to gwosc.org.

Usage:
    python scripts/curate_anomaly_gallery.py \
        --detector H1 --gps-start 1368975618 --gps-end 1369062018 \
        --stride 4 --top-n 50
"""
import argparse
import json
import uuid
from pathlib import Path

import torch
from gwpy.segments import DataQualityFlag

from model.model import load_model, GRAVITY_SPY_CLASSES
from model.gradcam import GradCAMExplainer
from model.ood import OODThreshold
from preprocessing.qtransform import preprocess

WEIGHTS_PATH = Path("model/weights/efficientnet_gravityspy.pt")
OOD_PATH = Path("model/weights/ood_threshold.json")
OUT_DIR = Path("backend/static/gallery")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["H1", "L1"])
    ap.add_argument("--gps-start", type=float, required=True)
    ap.add_argument("--gps-end", type=float, required=True)
    ap.add_argument("--stride", type=float, default=4.0, help="seconds between scan windows")
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(WEIGHTS_PATH), device=device)
    gradcam = GradCAMExplainer(model)
    ood = OODThreshold.load(str(OOD_PATH))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Restrict the scan to actual science-mode segments -- scanning
    # downtime just produces meaningless "anomalies."
    flag = DataQualityFlag.query(
        f"{args.detector}:DCS-ANALYSIS_READY_C01:1", args.gps_start, args.gps_end
    )

    candidates = []
    t = args.gps_start
    while t < args.gps_end:
        if not flag.active.intersects_segment((t, t + args.duration)):
            t += args.stride
            continue
        try:
            pre = preprocess("gps", detector=args.detector, gps_time=t, duration=args.duration)
        except Exception as e:
            print(f"  skip t={t}: {e}")
            t += args.stride
            continue

        import numpy as np
        tensor = torch.from_numpy(pre.image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        pred_idx, probs, logits, emb = model.predict(tensor)
        is_ood, score = ood.is_ood(logits)

        if is_ood:
            candidates.append((score, t, pre, int(pred_idx.item())))
            print(f"  t={t}: OOD score={score:.3f} (candidate #{len(candidates)})")

        t += args.stride

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[: args.top_n]

    manifest = []
    for score, t, pre, pred_idx in top:
        entry_id = uuid.uuid4().hex[:12]
        entry_dir = OUT_DIR / entry_id
        entry_dir.mkdir(parents=True, exist_ok=True)
        pre.as_pil().save(entry_dir / "spectrogram.png")

        tensor = torch.from_numpy(pre.image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        gradcam_bytes = gradcam.overlay_png_bytes(tensor, pre.image, class_idx=pred_idx)
        (entry_dir / "gradcam.png").write_bytes(gradcam_bytes)

        manifest.append({
            "id": entry_id,
            "detector": args.detector,
            "gps_time": t,
            "ood_score": score,
            "spectrogram_url": f"/static/gallery/{entry_id}/spectrogram.png",
            "gradcam_url": f"/static/gallery/{entry_id}/gradcam.png",
            "why_flagged": ood.interpretation(score),
            "nearest_known_class": GRAVITY_SPY_CLASSES[pred_idx],
        })

    existing_path = OUT_DIR / "anomalies.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else []
    existing.extend(manifest)
    existing_path.write_text(json.dumps(existing, indent=2))
    print(f"Added {len(manifest)} anomalies to {existing_path} (total {len(existing)})")


if __name__ == "__main__":
    main()
