"""
Inference service. Loads the model once at startup and exposes a single
`run_inference(...)` used by the /analyze route, regardless of which of
the 4 input formats the request came in as.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from model.model import GlitchClassifier, load_model, GRAVITY_SPY_CLASSES
from model.gradcam import GradCAMExplainer
from model.ood import OODThreshold
from preprocessing.qtransform import preprocess

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEIGHTS_PATH = Path(__file__).parent.parent / "model" / "weights" / "efficientnet_gravityspy.pt"
OOD_THRESHOLD_PATH = Path(__file__).parent.parent / "model" / "weights" / "ood_threshold.json"
RESULTS_DIR = Path(__file__).parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


class InferenceService:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GlitchClassifier | None = None
        self.classes = GRAVITY_SPY_CLASSES
        self.gradcam: GradCAMExplainer | None = None
        self.ood_threshold: OODThreshold | None = None

    def load(self):
        if not WEIGHTS_PATH.exists():
            raise RuntimeError(
                f"No trained weights found at {WEIGHTS_PATH}. Run "
                f"model/train.py first (see README Quickstart)."
            )
        self.model = load_model(str(WEIGHTS_PATH), device=self.device)
        self.gradcam = GradCAMExplainer(self.model)
        if OOD_THRESHOLD_PATH.exists():
            self.ood_threshold = OODThreshold.load(str(OOD_THRESHOLD_PATH))
        else:
            # Sensible fallback so the API doesn't hard-fail before OOD
            # calibration has been run; flag this loudly in logs.
            self.ood_threshold = OODThreshold(threshold=0.0)
            print("WARNING: no OOD threshold calibration found; using an "
                  "uncalibrated placeholder. Run scripts/calibrate_ood.py.")

    def image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        t = _normalize(t)
        return t.unsqueeze(0).to(self.device)

    def run(self, input_format: str, *, file=None, detector=None, gps_time=None, duration=1.0):
        if self.model is None:
            self.load()

        pre = preprocess(input_format, file=file, detector=detector, gps_time=gps_time, duration=duration)
        tensor = self.image_to_tensor(pre.image)

        pred_idx, probs, logits, emb = self.model.predict(tensor)
        pred_idx = int(pred_idx.item())
        probs_np = probs.squeeze(0).detach().cpu().numpy()

        is_ood, ood_score = self.ood_threshold.is_ood(logits)
        interpretation = self.ood_threshold.interpretation(ood_score)

        result_id = uuid.uuid4().hex[:12]
        result_dir = RESULTS_DIR / result_id
        result_dir.mkdir(parents=True, exist_ok=True)

        pre.as_pil().save(result_dir / "spectrogram.png")
        gradcam_bytes = self.gradcam.overlay_png_bytes(tensor, pre.image, class_idx=pred_idx)
        (result_dir / "gradcam.png").write_bytes(gradcam_bytes)

        payload = {
            "result_id": result_id,
            "predicted_class": self.classes[pred_idx],
            "confidence": float(probs_np[pred_idx]),
            "class_probabilities": {c: float(p) for c, p in zip(self.classes, probs_np)},
            "ood_score": ood_score,
            "ood_flagged": is_ood,
            "ood_interpretation": interpretation,
            "spectrogram_url": f"/static/results/{result_id}/spectrogram.png",
            "gradcam_url": f"/static/results/{result_id}/gradcam.png",
            "detector": pre.detector,
            "sample_rate": pre.sample_rate,
            "input_format": pre.source_format,
            "warnings": pre.warnings,
        }

        import json
        (result_dir / "meta.json").write_text(json.dumps(payload, indent=2))
        return payload


inference_service = InferenceService()
