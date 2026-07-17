import numpy as np
import torch

from model.gradcam import GradCAMExplainer
from model.model import NUM_CLASSES, GlitchClassifier


def test_explain_returns_normalized_cam_and_valid_class_idx():
    model = GlitchClassifier(pretrained=False)
    gradcam = GradCAMExplainer(model)
    x = torch.randn(1, 3, 224, 224)

    result = gradcam.explain(x)
    assert 0 <= result["class_idx"] < NUM_CLASSES
    assert result["cam"].min() >= 0.0
    assert result["cam"].max() <= 1.0 + 1e-5


def test_explain_respects_explicit_class_idx():
    model = GlitchClassifier(pretrained=False)
    gradcam = GradCAMExplainer(model)
    x = torch.randn(1, 3, 224, 224)

    result = gradcam.explain(x, class_idx=5)
    assert result["class_idx"] == 5


def test_overlay_png_bytes_produces_valid_png():
    model = GlitchClassifier(pretrained=False)
    gradcam = GradCAMExplainer(model)
    x = torch.randn(1, 3, 224, 224)
    original = (np.random.rand(224, 224, 3) * 255).astype("uint8")

    png_bytes = gradcam.overlay_png_bytes(x, original)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
    assert len(png_bytes) > 100
