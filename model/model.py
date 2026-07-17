"""
EfficientNet-based glitch classifier, fine-tuned on Gravity Spy.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# The 22 Gravity Spy classes (v1.1 taxonomy). Adjust if your dataset
# version differs -- keep this list in sync with the label encoder used
# in dataset.py / train.py.
GRAVITY_SPY_CLASSES = [
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch", "None_of_the_Above",
    "Paired_Doves", "Power_Line", "Repeating_Blips", "Scattered_Light",
    "Scratchy", "Tomte", "Violin_Mode", "Wandering_Line", "Whistle",
]
NUM_CLASSES = len(GRAVITY_SPY_CLASSES)


class GlitchClassifier(nn.Module):
    """EfficientNet-B0 backbone + linear head over Gravity Spy classes.

    Exposes `.features` (the conv backbone) and `.embedding(x)` so that
    both GradCAM (needs a conv layer to hook) and Mahalanobis-distance OOD
    detection (needs pre-logit embeddings) have what they need.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features          # conv stack (GradCAM target)
        self.avgpool = backbone.avgpool
        embed_dim = backbone.classifier[1].in_features
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        return self.classifier(self.dropout(emb))

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """Returns (predicted_class_idx, softmax_probs, logits, embedding)."""
        self.eval()
        emb = self.embedding(x)
        logits = self.classifier(emb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        return pred, probs, logits, emb


def load_model(weights_path: str, device: str = "cpu") -> GlitchClassifier:
    model = GlitchClassifier(pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.to(device)
    model.eval()
    return model
