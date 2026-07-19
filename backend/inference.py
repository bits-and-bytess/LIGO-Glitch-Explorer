"""
Inference service. Loads the model once at startup and exposes a single
`run_inference(...)` used by the /analyze route, regardless of which of
the 4 input formats the request came in as.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision import transforms

from model.gradcam import GradCAMExplainer
from model.model import GRAVITY_SPY_CLASSES, GlitchClassifier, load_model
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
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[GlitchClassifier] = None
        self.classes = GRAVITY_SPY_CLASSES
        self.ood_threshold: Optional[OODThreshold] = None

    def load(self):
        if not WEIGHTS_PATH.exists():
            raise RuntimeError(
                f"No trained weights found at {WEIGHTS_PATH}. Run "
                f"model/train.py first (see README Quickstart)."
            )
        self.model = load_model(str(WEIGHTS_PATH), device=self.device)
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
        # Built fresh per request and torn down immediately after: torchcam's
        # hook, once attached, intercepts EVERY subsequent forward pass
        # through the target layer -- including the next request's
        # model.predict() call above, which runs under @torch.no_grad().
        # A hook still attached from a prior request crashes that no-grad
        # forward pass with "cannot register a hook on a tensor that
        # doesn't require gradient." Explicit removal (not reliance on
        # __del__/GC timing) is what actually prevents that.
        gradcam = GradCAMExplainer(self.model)
        try:
            gradcam_bytes = gradcam.overlay_png_bytes(tensor, pre.image, class_idx=pred_idx)
        finally:
            gradcam.cam_extractor.remove_hooks()
        (result_dir / "gradcam.png").write_bytes(gradcam_bytes)

        payload = {
            "result_id": result_id,
            "predicted_class": self.classes[pred_idx],
            "confidence": float(probs_np[pred_idx]),
            "class_probabilities": {c: float(p) for c, p in zip(self.classes, probs_np)},
            "ood_score": ood_score,
            "ood_flagged": is_ood,
            "ood_threshold": self.ood_threshold.threshold,
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
