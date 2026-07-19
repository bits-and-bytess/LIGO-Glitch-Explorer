"""
Anomaly gallery router.

Serves the curated set of OOD-flagged O4 signals, pre-computed offline
(see scripts/curate_anomaly_gallery.py) rather than scanning live data on
every request.
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/gallery", tags=["gallery"])

GALLERY_DATA_PATH = Path(__file__).parent.parent / "static" / "gallery" / "anomalies.json"


def _load_gallery():
    if not GALLERY_DATA_PATH.exists():
        raise HTTPException(
            503,
            "Anomaly gallery not curated yet. Run "
            "scripts/curate_anomaly_gallery.py against O4 data first.",
        )
    return json.loads(GALLERY_DATA_PATH.read_text())


@router.get("")
async def list_anomalies(
    detector: Optional[str] = Query(None, pattern="^(H1|L1)$"),
    min_ood_score: Optional[float] = None,
    gps_start: Optional[float] = None,
    gps_end: Optional[float] = None,
):
    entries = _load_gallery()
    if detector:
        entries = [e for e in entries if e["detector"] == detector]
    if min_ood_score is not None:
        entries = [e for e in entries if e["ood_score"] >= min_ood_score]
    if gps_start is not None:
        entries = [e for e in entries if e["gps_time"] >= gps_start]
    if gps_end is not None:
        entries = [e for e in entries if e["gps_time"] <= gps_end]
    entries.sort(key=lambda e: e["ood_score"], reverse=True)
    return entries
