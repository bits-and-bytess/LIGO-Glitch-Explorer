"""
Calibrate the energy-score OOD threshold on held-out in-distribution
validation data (weeks 6-7 deliverable).

Usage:
    python scripts/calibrate_ood.py --data data/processed --percentile 95
    python scripts/calibrate_ood.py --data data/processed --weights model/weights/foo.pt --out model/weights/foo_ood.json
"""
import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/calibrate_ood.py` (not just
# `python -m scripts.calibrate_ood`) regardless of current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.dataset import get_dataloaders
from model.model import load_model
from model.ood import OODThreshold

DEFAULT_WEIGHTS_PATH = "model/weights/efficientnet_gravityspy.pt"
DEFAULT_OUT_PATH = "model/weights/ood_threshold.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH)
    ap.add_argument("--percentile", type=float, default=95.0,
                     help="in-distribution energy-score percentile to set as the cutoff")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device=device)
    _, val_loader, _, _ = get_dataloaders(args.data, num_workers=args.num_workers)

    all_logits = []
    with torch.no_grad():
        for images, _ in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)

    threshold = OODThreshold.fit(all_logits, temperature=args.temperature, percentile=args.percentile)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    threshold.save(args.out)
    print(f"Calibrated OOD threshold = {threshold.threshold:.4f} "
          f"(percentile={args.percentile}, temperature={args.temperature})")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
