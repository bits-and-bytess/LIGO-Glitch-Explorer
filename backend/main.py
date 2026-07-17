from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.routers import analyze, library, gallery
from backend.inference import inference_service

app = FastAPI(
    title="LIGO Glitch Explorer API",
    description="Classification, GradCAM explanation, and OOD flagging for "
                "LIGO detector-noise glitches.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel domain before going public
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(analyze.router)
app.include_router(library.router)
app.include_router(gallery.router)


@app.on_event("startup")
async def startup():
    try:
        inference_service.load()
        print("Model loaded successfully.")
    except RuntimeError as e:
        # Don't crash the whole API if weights aren't trained yet -- let
        # /library and /gallery (which serve pre-generated static data)
        # keep working, and fail /analyze requests with a clear message.
        print(f"WARNING: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inference_service.model is not None}
