# LIGO Glitch Explorer backend
#
# Works for both Hugging Face Spaces (Docker SDK) and Render:
#  - HF Spaces: place this at repo root (or set app_file in README metadata),
#    Spaces will build and run it, exposing port 7860 by default.
#  - Render: "Docker" runtime, no changes needed; Render sets $PORT for you.
#
# Model weights (model/weights/*.pt, ood_threshold.json) and pre-generated
# static assets (backend/static/library, backend/static/gallery) are NOT
# baked into the image here -- they're large binary artifacts that don't
# belong in git history. Options:
#   1. Bake them in with an extra COPY step once you have them (uncomment below)
#   2. Mount them as a volume
#   3. Pull them from a release asset / object storage on container start
# Pick whichever matches your hosting choice; (1) is simplest for a single
# free-tier deploy.

FROM python:3.11-slim

# System deps: gwpy/lalsuite-adjacent packages sometimes need these for
# building from source; keep the image lean but avoid silent build failures.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY model/ ./model/
COPY preprocessing/ ./preprocessing/

# Uncomment once you have trained weights + calibrated OOD threshold to ship:
# COPY model/weights/ ./model/weights/
# COPY backend/static/library/ ./backend/static/library/
# COPY backend/static/gallery/ ./backend/static/gallery/

# HF Spaces expects the app to listen on $PORT (defaults to 7860);
# Render also injects $PORT. Fall back to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
