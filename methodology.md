# Methodology

## Dataset
- **Training/reference library:** Gravity Spy training set (Bahaadini et al.,
  Zenodo), ~10,000 citizen-science-labeled glitches across 22 classes.
- **Anomaly gallery source:** public O4 strain data from GWOSC, scanned
  offline with `scripts/curate_anomaly_gallery.py`.

## Pipeline
1. Strain segment (from HDF5, GWOSC GPS query, or CSV) → gwpy Q-transform
   (qrange 4–64, frange 10–2048 Hz, whitened) → rasterized to a 224×224 RGB
   image. This is the single contract in `preprocessing/qtransform.py` that
   every input format must satisfy before reaching the model.
2. Pre-made spectrogram images skip step 1 entirely; dimensions/aspect
   ratio are validated and a warning surfaces if GradCAM is likely to be
   less meaningful on that input.
3. EfficientNet-B0 (ImageNet-pretrained) fine-tuned on Gravity Spy
   (`model/train.py`): 2-phase schedule, classifier head only for the
   first few epochs, then full backbone fine-tuning.
4. GradCAM (`model/gradcam.py`, via torchcam) computed on the final conv
   block for every inference and for representative library examples.
5. Energy-based OOD score (`model/ood.py`) on output logits, threshold
   calibrated on held-out validation data (`scripts/calibrate_ood.py`).
   Mahalanobis-distance-on-embeddings is implemented as a stricter,
   optional upgrade.

## Known limitations
- H1/L1 only; Virgo (V1) and KAGRA are out of scope for this iteration
  because their noise profiles would need separate training data.
- Gravity Spy is H1-dominated, so H1 accuracy is expected to exceed L1;
  cross-detector generalization is an explicit open experiment, not a
  guarantee.
- CSV sample-rate inference is unreliable on irregularly-sampled data —
  the pipeline detects and warns on this, but can't fully correct for it.
- An OOD flag means "doesn't match a known class well," not "confirmed
  novel astrophysical or instrumental phenomenon." The anomaly gallery is
  a curation starting point for human review, not a discovery claim.

## Reproducing results
Every `/analyze` result gets a permalink (`/analyze/{result_id}`) backed
by `backend/static/results/{id}/meta.json`, `spectrogram.png`, and
`gradcam.png` — enough to cite or re-inspect a specific analysis later.
