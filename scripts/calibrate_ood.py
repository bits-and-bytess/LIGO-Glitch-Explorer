"""
Calibrate the energy-score OOD threshold on held-out in-distribution
validation data (weeks 6-7 deliverable).

Usage:
    python scripts/calibrate_ood.py --data data/processed --percentile 95
"""
import argparse
from pathlib import Path

import torch

from model.dataset import get_dataloaders
from model.model import load_model
from model.ood import OODThreshold

WEIGHTS_PATH = Path("model/weights/efficientnet_gravityspy.pt")
OUT_PATH = Path("model/weights/ood_threshold.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--percentile", type=float, default=95.0,
                     help="in-distribution energy-score percentile to set as the cutoff")
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(WEIGHTS_PATH), device=device)
    _, val_loader, _, _ = get_dataloaders(args.data)

    all_logits = []
    with torch.no_grad():
        for images, _ in val_loader:
            logits = model(images.to(device))
            all_logits.append(logits.cpu())
    all_logits = torch.cat(all_logits, dim=0)

    threshold = OODThreshold.fit(all_logits, temperature=args.temperature, percentile=args.percentile)
    threshold.save(str(OUT_PATH))
    print(f"Calibrated OOD threshold = {threshold.threshold:.4f} "
          f"(percentile={args.percentile}, temperature={args.temperature})")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
