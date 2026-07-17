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

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data (run on a machine with internet access to zenodo.org)
python data/download_gravityspy.py --out data/raw

# 3. Preprocess to Q-transform spectrograms (the unified image contract)
python preprocessing/qtransform.py --in data/raw --out data/processed

# 4. Train baseline classifier
python model/train.py --data data/processed --epochs 20 --out model/weights/efficientnet_gravityspy.pt

# 5. Run the backend
cd backend && uvicorn main:app --reload --port 8000

# 6. Run the frontend
cd frontend && npm install && npm run dev
```

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
