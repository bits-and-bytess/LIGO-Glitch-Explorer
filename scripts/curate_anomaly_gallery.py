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
import sys
import uuid
from pathlib import Path

# Allow running as `python scripts/curate_anomaly_gallery.py` regardless
# of current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from gwpy.segments import DataQualityFlag

from model.gradcam import GradCAMExplainer
from model.model import GRAVITY_SPY_CLASSES, load_model
from model.ood import OODThreshold
from preprocessing.qtransform import preprocess

DEFAULT_WEIGHTS_PATH = "model/weights/efficientnet_gravityspy.pt"
DEFAULT_OOD_PATH = "model/weights/ood_threshold.json"
DEFAULT_OUT_DIR = "backend/static/gallery"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=["H1", "L1"])
    ap.add_argument("--gps-start", type=float, required=True)
    ap.add_argument("--gps-end", type=float, required=True)
    ap.add_argument("--stride", type=float, default=4.0, help="seconds between scan windows")
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--ood-threshold-path", default=DEFAULT_OOD_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device=device)
    ood = OODThreshold.load(args.ood_threshold_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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

        tensor = torch.from_numpy(pre.image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        pred_idx, probs, logits, emb = model.predict(tensor)
        is_ood, score = ood.is_ood(logits)

        if is_ood:
            candidates.append((score, t, pre, int(pred_idx.item())))
            print(f"  t={t}: OOD score={score:.3f} (candidate #{len(candidates)})")

        t += args.stride

    candidates.sort(key=lambda c: c[0], reverse=True)
    top = candidates[: args.top_n]

    # Constructed only now, after scanning is done: torchcam's hook,
    # once attached, intercepts EVERY subsequent forward pass through the
    # target layer, including the plain model.predict() calls above (which
    # run under @torch.no_grad() and crash if a GradCAM hook is already
    # attached). Building GradCAM only when it's actually needed, for the
    # much smaller top-N set, avoids that entirely.
    gradcam = GradCAMExplainer(model)

    manifest = []
    for score, t, pre, pred_idx in top:
        entry_id = uuid.uuid4().hex[:12]
        entry_dir = out_dir / entry_id
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

    existing_path = out_dir / "anomalies.json"
    existing = json.loads(existing_path.read_text()) if existing_path.exists() else []
    existing.extend(manifest)
    existing_path.write_text(json.dumps(existing, indent=2))
    print(f"Added {len(manifest)} anomalies to {existing_path} (total {len(existing)})")


if __name__ == "__main__":
    main()
