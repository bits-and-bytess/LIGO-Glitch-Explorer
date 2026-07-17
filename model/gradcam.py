"""
GradCAM overlays for the glitch classifier, via torchcam.

Produces both the raw CAM (for programmatic use, e.g. computing overlap
between classes) and a ready-to-display PNG overlay on the input
spectrogram.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torchcam.methods import GradCAM as TorchCamGradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

from model.model import GlitchClassifier


class GradCAMExplainer:
    """Wraps torchcam's GradCAM around the last conv block of the backbone."""

    def __init__(self, model: GlitchClassifier, target_layer: str = "features.8"):
        self.model = model
        self.model.eval()
        # features.8 is the final conv block in EfficientNet-B0's feature
        # extractor -- highest-level spatial features, standard GradCAM target.
        self.cam_extractor = TorchCamGradCAM(model, target_layer=target_layer)

    def explain(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """
        input_tensor: (1, 3, 224, 224) normalized tensor, requires_grad not needed
        class_idx: which class to explain; defaults to the predicted class

        Returns: dict with keys
            cam: (H, W) float32 array in [0, 1], raw activation map
            class_idx: int, class explained
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        cams = self.cam_extractor(class_idx, logits)
        cam = cams[0].squeeze(0).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return {"cam": cam, "class_idx": class_idx}

    def overlay_png_bytes(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                           class_idx: int | None = None, alpha: float = 0.5) -> bytes:
        """Returns PNG bytes of the spectrogram with the GradCAM heatmap overlaid."""
        result = self.explain(input_tensor, class_idx)
        cam_img = to_pil_image(torch.from_numpy(result["cam"]).unsqueeze(0), mode="F")

        base_img = Image.fromarray(original_image).convert("RGB")
        overlay = overlay_mask(base_img, cam_img, colormap="jet", alpha=alpha)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        return buf.getvalue()

    def __del__(self):
        # torchcam registers forward/backward hooks; clean up explicitly
        # since these explainers are often created per-request in the API.
        try:
            self.cam_extractor.remove_hooks()
        except Exception:
            pass
