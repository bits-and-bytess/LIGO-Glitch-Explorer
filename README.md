# LIGO Glitch Explorer

A public tool for classifying, explaining, and surfacing anomalies in
gravitational-wave detector noise. Built on the Gravity Spy dataset and
GWOSC public strain data.

## Status / how to read this repo

This repo is a working scaffold for the full 10-week project in your spec.
It is organized so each week's deliverable maps to a folder:

| Weeks | Deliverable                  | Folder                          |
|-------|-------------------------------|----------------------------------|
| 1–2   | Data pipeline                 | `data/`, `preprocessing/`        |
| 3–5   | Baseline classifier           | `model/train.py`, `model/model.py` |
| 6–7   | GradCAM + OOD                 | `model/gradcam.py`, `model/ood.py` |
| 8–9   | Website                       | `backend/`, `frontend/`          |
| 10    | Polish + publish              | `methodology.md`, this README    |

**Important network note:** this scaffold was built in a sandboxed
environment that cannot reach `gwosc.org`, `losc.ligo.org`, or `zenodo.org`.
The download/query code is real and correct, but you need to actually run
`data/download_gravityspy.py` and exercise the GWOSC API path on a machine
with normal internet access. Everything else (preprocessing math, model,
GradCAM, OOD scoring, backend, frontend) runs and can be tested locally
right now with synthetic or your own data.

## Quickstart

Requires **Python 3.9+**. (An earlier version of this code used `X | None`
union-type syntax that only works on Python 3.10+ without care; this
was found and fixed after failing on a real Python 3.9 environment --
see CI, which now runs the test suite against both 3.9 and 3.12.)

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data (run on a machine with internet access -- not this scaffold's sandbox)
#    Pulls trainingset_v1d1_metadata.csv + trainingsetv1d1.tar.gz (~5.5GB)
#    from the verified Zenodo record https://zenodo.org/records/1476551
python data/download_gravityspy.py --out data/raw

#    Converts the raw tarball into data/processed/{train,val,test}/<class>/*.png:
#    crops out the matplotlib axes (per the dataset's own documented crop box),
#    picks one duration per sample (default 1.0s), resizes to 224x224.
python scripts/build_dataset.py \
    --metadata data/raw/trainingset_v1d1_metadata.csv \
    --tarball data/raw/trainingsetv1d1.tar.gz \
    --out data/processed

# 3. Train baseline classifier
python model/train.py --data data/processed --epochs 20 --out model/weights/efficientnet_gravityspy.pt

# 4. Run the backend (from repo root -- NOT `cd backend` first, since
#    backend/main.py uses absolute imports like `from backend.inference import ...`
#    that only resolve with the repo root on sys.path)
uvicorn backend.main:app --reload --port 8000

# 5. Run the frontend
cd frontend && npm install && npm run dev
```

Note: an earlier version of this README described a single preprocessing
step directly from the download. That was wrong about the actual file
layout -- see `data/download_gravityspy.py` and `scripts/build_dataset.py`
docstrings for the corrected two-step pipeline, verified against the live
Zenodo record.

## Testing & CI

```bash
pip install -r requirements-dev.txt --break-system-packages
pytest -v
ruff check .
```

GitHub Actions (`.github/workflows/ci.yml`) runs the backend test suite +
lint and the frontend production build on every push/PR to `main`. The
test suite is deliberately scoped to things that don't need gwpy/network
access or trained weights (model architecture, OOD math, the image/CSV
preprocessing branches) so CI stays fast and doesn't depend on GWOSC being
reachable from the runner.

## Deployment

**Backend** (Hugging Face Spaces or Render, both free-tier friendly):
```bash
docker build -t ligo-glitch-backend .
docker run -p 8000:8000 ligo-glitch-backend
```
The `Dockerfile` does **not** bake in model weights or pre-generated
static assets by default (they're large binaries that don't belong in
git history) -- see the comments at the top of the `Dockerfile` for the
three ways to get them into the running container (bake in, volume mount,
or pull from object storage on startup).

- *Hugging Face Spaces:* create a Space with the Docker SDK, push this
  repo to it (or link it), Spaces builds from the `Dockerfile` automatically.
- *Render:* new Web Service → Docker runtime → point at this repo. Render
  sets `$PORT` for you; the Dockerfile already respects it.

**Frontend** (Vercel):
```bash
cd frontend
vercel deploy
```
Set the `VITE_API_BASE` environment variable in your Vercel project
settings to your deployed backend's URL (see `frontend/.env.example`).
`vercel.json` includes the SPA rewrite rule React Router needs for direct
URL loads (e.g. a shared `/analyze/{result_id}` permalink).

## The one contract that holds the whole system together

Every input path (HDF5 strain, GPS+detector, raw image, CSV timeseries)
must produce **the same artifact** before it touches the model:

> a normalized 224×224 RGB Q-transform spectrogram image, plus a
> `PreprocessResult` metadata object (sample rate, detector, warnings).

This is enforced by `preprocessing/qtransform.py`'s `preprocess(...)`
function, which is the single entry point the FastAPI backend calls
regardless of `input_format`. Keeping this contract narrow is what lets
the model, GradCAM, and OOD scoring stay completely input-format-agnostic.

See `methodology.md` for the model/architecture writeup you'll expand
into your public Methodology page.
