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

   Important internal detail: whitening estimates a noise PSD via Welch's
   method, which needs meaningfully more data than the displayed window
   itself. The pipeline therefore keeps a longer buffer around the
   requested time (target: 4s padding each side) and uses gwpy's `outseg`
   to crop only the *output* after whitening, rather than cropping first.
   Below ~1s of padding, whitening is disabled entirely and the Q-transform
   falls back to unwhitened output rather than crashing -- this is
   surfaced to the user as an explicit warning, not silently.

   Also worth knowing: gwpy auto-narrows the requested frequency range
   based on the interaction between qrange and the signal's sample rate --
   at 4096 Hz with qrange (4, 64), the nominal 2048 Hz upper bound gets
   reset to roughly 1291 Hz. This is gwpy's own safety behavior, not a bug,
   but it means the analyzed band is sometimes narrower than the nominal
   configuration.

   Strain data can also contain NaNs wherever a detector wasn't in science
   mode or failed data-quality checks. The pipeline checks the fraction of
   NaNs specifically within the requested output window: below 50% it
   fills gaps and proceeds with a warning; above 50% it refuses rather
   than return a classification built on mostly-absent data.
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
