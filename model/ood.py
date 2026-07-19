"""
Out-of-distribution detection.

Primary method: energy-based score on output logits (Liu et al. 2020,
"Energy-based Out-of-distribution Detection"). Cheap, needs no extra
training, works directly off logits the classifier already produces.

Optional upgrade: Mahalanobis distance in embedding space, which tends to
catch OOD signals that "look like" a known class in logit-space but sit
far from that class's embedding cluster. Requires fitting per-class
Gaussians on training embeddings first (fit_mahalanobis below).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------
# Energy score
# --------------------------------------------------------------------------
def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Lower energy = more in-distribution (model is confidently peaked on
    one class). Higher energy = more OOD-like (flat / low-magnitude logits).

    E(x) = -T * logsumexp(logits / T)

    Returns a 1D tensor of energy scores, one per input in the batch.
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


@dataclass
class OODThreshold:
    threshold: float
    temperature: float = 1.0

    @classmethod
    def fit(cls, val_logits: torch.Tensor, temperature: float = 1.0,
            percentile: float = 95.0) -> "OODThreshold":
        """
        Fit the threshold on held-out in-distribution validation data:
        set it at the given percentile of in-distribution energy scores,
        so ~ (100 - percentile)% of legitimate known-class val signals
        would be (incorrectly) flagged -- tune `percentile` against your
        actual false-positive tolerance.
        """
        scores = energy_score(val_logits, temperature).cpu().numpy()
        threshold = float(np.percentile(scores, percentile))
        return cls(threshold=threshold, temperature=temperature)

    def save(self, path: str):
        Path(path).write_text(json.dumps({"threshold": self.threshold, "temperature": self.temperature}))

    @classmethod
    def load(cls, path: str) -> "OODThreshold":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def is_ood(self, logits: torch.Tensor) -> tuple[bool, float]:
        score = float(energy_score(logits, self.temperature).item())
        return score > self.threshold, score

    def interpretation(self, score: float) -> str:
        margin = score - self.threshold
        if margin <= 0:
            return (
                "Signal's energy score is within the range seen for known "
                "glitch classes -- the classification above is likely reliable."
            )
        elif margin < 0.5 * abs(self.threshold):
            return (
                "Signal's energy score is somewhat above the known-class "
                "range. The predicted class may still be right, but treat "
                "the confidence with some caution."
            )
        else:
            return (
                "Signal's energy score is well outside the range seen for "
                "any known glitch class. This looks meaningfully different "
                "from the training distribution and is a candidate for the "
                "anomaly gallery."
            )


# --------------------------------------------------------------------------
# Optional upgrade: Mahalanobis distance on embeddings
# --------------------------------------------------------------------------
@dataclass
class MahalanobisOOD:
    class_means: np.ndarray       # (num_classes, embed_dim)
    shared_precision: np.ndarray  # (embed_dim, embed_dim), inverse of pooled covariance
    threshold: Optional[float] = None

    @classmethod
    def fit(cls, embeddings: np.ndarray, labels: np.ndarray, num_classes: int) -> "MahalanobisOOD":
        embed_dim = embeddings.shape[1]
        means = np.zeros((num_classes, embed_dim))
        centered_all = []
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            means[c] = embeddings[mask].mean(axis=0)
            centered_all.append(embeddings[mask] - means[c])
        centered_all = np.concatenate(centered_all, axis=0)
        cov = np.cov(centered_all, rowvar=False) + 1e-6 * np.eye(embed_dim)
        precision = np.linalg.inv(cov)
        return cls(class_means=means, shared_precision=precision)

    def score(self, embedding: np.ndarray) -> float:
        """Minimum Mahalanobis distance to any class mean (smaller = more in-distribution)."""
        diffs = self.class_means - embedding[None, :]
        dists = np.einsum("ij,jk,ik->i", diffs, self.shared_precision, diffs)
        return float(np.min(dists))

    def fit_threshold(self, val_embeddings: np.ndarray, percentile: float = 95.0):
        scores = [self.score(e) for e in val_embeddings]
        self.threshold = float(np.percentile(scores, percentile))

    def is_ood(self, embedding: np.ndarray) -> tuple[bool, float]:
        s = self.score(embedding)
        return (self.threshold is not None and s > self.threshold), s

    def save(self, path: str):
        np.savez(path, class_means=self.class_means, shared_precision=self.shared_precision,
                  threshold=self.threshold if self.threshold is not None else -1.0)

    @classmethod
    def load(cls, path: str) -> "MahalanobisOOD":
        data = np.load(path)
        thr = float(data["threshold"])
        return cls(class_means=data["class_means"], shared_precision=data["shared_precision"],
                    threshold=None if thr == -1.0 else thr)
