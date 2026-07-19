from typing import Optional

from pydantic import BaseModel, Field


class AnalyzeGPSRequest(BaseModel):
    gps_time: float
    detector: str = Field(..., pattern="^(H1|L1)$")
    duration: float = 1.0


class AnalyzeResponse(BaseModel):
    result_id: str                 # for the shareable permalink
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    ood_score: float
    ood_flagged: bool
    ood_threshold: float
    ood_interpretation: str
    spectrogram_url: str
    gradcam_url: str
    detector: Optional[str] = None
    sample_rate: Optional[float] = None
    input_format: str
    warnings: list[str] = []


class GlitchClassSummary(BaseModel):
    name: str
    description: str
    physical_origin: Optional[str]
    frequency_range_hz: tuple[float, float]
    typical_duration_s: tuple[float, float]
    example_spectrogram_url: str
    example_gradcam_url: str
    similar_classes: list[str] = []


class AnomalyEntry(BaseModel):
    id: str
    detector: str
    gps_time: float
    ood_score: float
    spectrogram_url: str
    gradcam_url: str
    why_flagged: str
    nearest_known_class: Optional[str] = None
