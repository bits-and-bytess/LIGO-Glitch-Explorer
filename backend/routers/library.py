"""
Glitch library router.

Per the spec: "Pre-run results stored as static JSON + images so the
library pages load fast without hitting the model." This router just
reads that pre-generated JSON -- it never runs inference. Generate the
JSON with scripts/generate_library_assets.py.
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/library", tags=["library"])

LIBRARY_DATA_PATH = Path(__file__).parent.parent / "static" / "library" / "classes.json"


def _load_library():
    if not LIBRARY_DATA_PATH.exists():
        raise HTTPException(
            503,
            "Library assets not generated yet. Run "
            "scripts/generate_library_assets.py after training the model.",
        )
    return json.loads(LIBRARY_DATA_PATH.read_text())


@router.get("")
async def list_classes(search: Optional[str] = None):
    classes = _load_library()
    if search:
        s = search.lower()
        classes = [c for c in classes if s in c["name"].lower() or s in c["description"].lower()]
    return classes


@router.get("/{class_name}")
async def get_class(class_name: str):
    classes = _load_library()
    match = next((c for c in classes if c["name"] == class_name), None)
    if not match:
        raise HTTPException(404, f"Unknown glitch class '{class_name}'")
    return match
