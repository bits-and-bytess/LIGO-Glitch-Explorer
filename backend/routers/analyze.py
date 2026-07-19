import io
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.inference import inference_service
from backend.schemas import AnalyzeResponse
from preprocessing.qtransform import VALID_DURATIONS, PreprocessError

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    input_format: str = Form(...),   # "hdf5" | "image" | "csv"
    detector: Optional[str] = Form(None),
    duration: float = Form(1.0),
):
    if input_format not in ("hdf5", "image", "csv"):
        raise HTTPException(400, f"input_format must be hdf5, image, or csv for file upload (got {input_format})")
    if duration not in VALID_DURATIONS:
        raise HTTPException(400, f"duration must be one of {VALID_DURATIONS}")

    contents = await file.read()
    buf = io.BytesIO(contents)

    try:
        result = inference_service.run(
            input_format, file=buf, detector=detector, duration=duration
        )
    except PreprocessError as e:
        raise HTTPException(422, str(e))
    return result


@router.post("/gps", response_model=AnalyzeResponse)
async def analyze_gps(
    gps_time: float = Form(...),
    detector: str = Form(...),
    duration: float = Form(1.0),
):
    if detector not in ("H1", "L1"):
        raise HTTPException(400, "detector must be H1 or L1")
    if duration not in VALID_DURATIONS:
        raise HTTPException(400, f"duration must be one of {VALID_DURATIONS}")

    try:
        result = inference_service.run(
            "gps", detector=detector, gps_time=gps_time, duration=duration
        )
    except PreprocessError as e:
        raise HTTPException(422, str(e))
    return result


@router.get("/{result_id}", response_model=AnalyzeResponse)
async def get_result(result_id: str):
    """Fetch a previously computed result by its shareable permalink id."""
    import json
    from pathlib import Path

    result_meta_path = Path(__file__).parent.parent / "static" / "results" / result_id / "meta.json"
    if not result_meta_path.exists():
        raise HTTPException(404, "Result not found. Results are only cached temporarily -- "
                                  "re-run the analysis if this is an old link.")
    return json.loads(result_meta_path.read_text())
