"""
Unified preprocessing contract.

Every input format defined in the spec (hdf5, gps, image, csv) funnels
through `preprocess(...)` below and comes out the other side as the same
artifact:

    PreprocessResult(
        image: np.ndarray  # (224, 224, 3) uint8, RGB, Q-transform colormap
        sample_rate: float | None
        detector: str | None
        warnings: list[str]
        source_format: str
    )

This is the single seam between "whatever the user gave us" and "what the
CNN expects." The FastAPI backend should never touch gwpy, h5py, or PIL
directly outside of this module -- it just calls preprocess() and gets a
tensor-ready image back.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---- fixed / internal parameters (spec: "not exposed to end users") ----
Q_RANGE = (4, 64)
FREQUENCY_RANGE = (10, 2048)  # Hz
IMAGE_SIZE = (224, 224)
EXPECTED_SAMPLE_RATES = (4096, 16384)
VALID_DETECTORS = ("H1", "L1")
DEFAULT_DURATION = 1.0  # seconds
VALID_DURATIONS = (0.5, 1.0, 2.0, 4.0)


@dataclass
class PreprocessResult:
    image: np.ndarray
    sample_rate: Optional[float]
    detector: Optional[str]
    source_format: str
    warnings: list = field(default_factory=list)

    def as_pil(self):
        from PIL import Image
        return Image.fromarray(self.image, mode="RGB")


class PreprocessError(ValueError):
    """Raised for unrecoverable input problems (bad file, no data, etc)."""


# --------------------------------------------------------------------------
# Format 1: HDF5 strain file
# --------------------------------------------------------------------------
def _from_hdf5(path_or_buffer, detector: Optional[str], duration: float) -> PreprocessResult:
    import h5py
    from gwpy.timeseries import TimeSeries

    warnings = []
    with h5py.File(path_or_buffer, "r") as f:
        # GWOSC hdf5 strain files store the channel under
        # /strain/Strain with sample_rate in attrs, and meta/Detector.
        if "strain" not in f or "Strain" not in f["strain"]:
            raise PreprocessError(
                "HDF5 file does not contain the expected /strain/Strain "
                "dataset. Is this a GWOSC strain file?"
            )
        strain_ds = f["strain/Strain"]
        dt = strain_ds.attrs.get("Xspacing")
        sample_rate = round(1.0 / dt) if dt else None
        meta_detector = None
        if "meta" in f and "Detector" in f["meta"]:
            raw = f["meta"]["Detector"][()]
            meta_detector = raw.decode() if isinstance(raw, bytes) else str(raw)

    resolved_detector = detector or meta_detector
    if resolved_detector not in VALID_DETECTORS:
        warnings.append(
            f"Detector '{resolved_detector}' not recognized/supported "
            f"(expected one of {VALID_DETECTORS}); proceeding but accuracy "
            f"may be degraded."
        )

    if sample_rate and sample_rate not in EXPECTED_SAMPLE_RATES:
        warnings.append(
            f"Sample rate {sample_rate} Hz is not one of the expected "
            f"values {EXPECTED_SAMPLE_RATES}; results may be unreliable."
        )

    ts = TimeSeries.read(path_or_buffer, format="hdf5.gwosc")
    ts = _center_crop(ts, duration, warnings)
    image = _qtransform_to_image(ts, warnings)

    return PreprocessResult(
        image=image,
        sample_rate=sample_rate,
        detector=resolved_detector,
        source_format="hdf5",
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Format 2: GPS time + detector -> pull from GWOSC
# --------------------------------------------------------------------------
def _from_gps(gps_time: float, detector: str, duration: float) -> PreprocessResult:
    from gwpy.segments import DataQualityFlag
    from gwpy.timeseries import TimeSeries

    warnings = []
    if detector not in VALID_DETECTORS:
        raise PreprocessError(
            f"detector must be one of {VALID_DETECTORS} for GPS input, "
            f"got '{detector}'"
        )

    pad = max(duration, 4.0)  # fetch a bit of margin for the Q-transform
    start, end = gps_time - pad, gps_time + pad

    # Fallback: confirm the window is in a science-mode segment before
    # fetching. If not, widen the search or flag it clearly rather than
    # silently returning noise-only / no data.
    flag_name = f"{detector}:DCS-ANALYSIS_READY_C01:1"
    try:
        science_segments = DataQualityFlag.query(flag_name, start, end)
        if not science_segments.active:
            warnings.append(
                f"Requested GPS time {gps_time} does not fall within a "
                f"known science-mode segment for {detector}. The detector "
                f"may have been down or in an engineering run; results "
                f"should be treated with caution."
            )
    except Exception as e:  # pragma: no cover - network/service dependent
        warnings.append(
            f"Could not verify science-mode segment ({e}); proceeding "
            f"without that check."
        )

    try:
        ts = TimeSeries.fetch_open_data(detector, start, end, cache=True)
    except Exception as e:
        raise PreprocessError(
            f"Failed to fetch GWOSC open data for {detector} at GPS "
            f"{gps_time}: {e}"
        )

    sample_rate = ts.sample_rate.value
    if sample_rate not in EXPECTED_SAMPLE_RATES:
        warnings.append(
            f"Fetched sample rate {sample_rate} Hz is unexpected; "
            f"expected one of {EXPECTED_SAMPLE_RATES}."
        )

    ts = _center_crop(ts, duration, warnings, center_gps=gps_time)
    image = _qtransform_to_image(ts, warnings)

    return PreprocessResult(
        image=image,
        sample_rate=sample_rate,
        detector=detector,
        source_format="gps",
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Format 3: pre-made spectrogram image (PNG/JPG) - skips gwpy entirely
# --------------------------------------------------------------------------
def _from_image(path_or_buffer) -> PreprocessResult:
    from PIL import Image

    warnings = []
    img = Image.open(path_or_buffer).convert("RGB")
    w, h = img.size
    expected_aspect = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    actual_aspect = w / h
    if abs(actual_aspect - expected_aspect) > 0.15:
        warnings.append(
            f"Uploaded image aspect ratio ({w}x{h}) does not match the "
            f"expected Q-transform format (~square). GradCAM output may "
            f"be less meaningful for this input since it wasn't produced "
            f"by our own preprocessing pipeline."
        )
    if min(w, h) < 64:
        warnings.append(
            "Uploaded image resolution is very low; classification "
            "confidence may be reduced."
        )

    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
    return PreprocessResult(
        image=np.array(img),
        sample_rate=None,
        detector=None,
        source_format="image",
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Format 4: raw timeseries CSV (time, strain columns)
# --------------------------------------------------------------------------
def _from_csv(path_or_buffer, detector: Optional[str], duration: float) -> PreprocessResult:
    import pandas as pd

    warnings = []
    df = pd.read_csv(path_or_buffer)
    cols_lower = {c.lower(): c for c in df.columns}
    time_col = next((cols_lower[c] for c in ("time", "t", "gps_time") if c in cols_lower), None)
    strain_col = next((cols_lower[c] for c in ("strain", "value", "amplitude") if c in cols_lower), None)

    if time_col is None or strain_col is None:
        raise PreprocessError(
            "CSV must contain a time column (time/t/gps_time) and a "
            "strain column (strain/value/amplitude)."
        )

    t = df[time_col].to_numpy(dtype=float)
    y = df[strain_col].to_numpy(dtype=float)
    if len(t) < 3:
        raise PreprocessError("CSV has too few samples to infer a sample rate.")

    dt_diffs = np.diff(t)
    if np.any(dt_diffs <= 0):
        raise PreprocessError("Time column is not monotonically increasing.")
    dt_median = np.median(dt_diffs)
    dt_std = np.std(dt_diffs)

    inferred_rate = 1.0 / dt_median
    # Irregular timestamps => sample rate inference is unreliable (spec
    # explicitly calls this out as the least reliable path).
    if dt_std / dt_median > 0.01:
        warnings.append(
            f"Time spacing is irregular (std/median = {dt_std/dt_median:.3f}); "
            f"inferred sample rate ({inferred_rate:.1f} Hz) may be wrong. "
            f"Consider resampling to a fixed rate before uploading."
        )

    nearest = min(EXPECTED_SAMPLE_RATES, key=lambda r: abs(r - inferred_rate))
    if abs(nearest - inferred_rate) / nearest > 0.05:
        warnings.append(
            f"Inferred sample rate {inferred_rate:.1f} Hz doesn't closely "
            f"match a standard LIGO rate {EXPECTED_SAMPLE_RATES}; using "
            f"the inferred rate as-is, but treat results with caution."
        )

    from gwpy.timeseries import TimeSeries

    ts = TimeSeries(y, sample_rate=inferred_rate, t0=t[0])
    ts = _center_crop(ts, duration, warnings)
    image = _qtransform_to_image(ts, warnings)

    return PreprocessResult(
        image=image,
        sample_rate=inferred_rate,
        detector=detector,
        source_format="csv",
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _center_crop(ts, duration: float, warnings: list, center_gps: Optional[float] = None):
    if duration not in VALID_DURATIONS:
        warnings.append(
            f"Requested duration {duration}s is not one of the suggested "
            f"options {VALID_DURATIONS}; using it anyway."
        )
    center = center_gps if center_gps is not None else (ts.t0.value + ts.duration.value / 2)
    half = duration / 2
    try:
        return ts.crop(center - half, center + half)
    except Exception:
        # not enough data on one side; clip to what's available
        warnings.append(
            "Requested time window extends beyond available data; "
            "cropped to the available segment."
        )
        return ts


def _qtransform_to_image(ts, warnings: list) -> np.ndarray:
    """Run the Q-transform and rasterize to a fixed-size RGB array.

    This is the ONE function that defines the "Q-transform colormap"
    contract every input path must match.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        qspec = ts.q_transform(
            qrange=Q_RANGE,
            frange=FREQUENCY_RANGE,
            whiten=True,
        )
    except Exception as e:
        raise PreprocessError(f"Q-transform failed: {e}")

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(
        qspec.value.T,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=(0, qspec.shape[0], 0, qspec.shape[1]),
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
    return np.array(img)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def preprocess(
    input_format: str,
    *,
    file=None,
    detector: Optional[str] = None,
    gps_time: Optional[float] = None,
    duration: float = DEFAULT_DURATION,
) -> PreprocessResult:
    """
    Single entry point for all four input formats.

    input_format: one of "hdf5", "gps", "image", "csv"
    file: file path or file-like object (required for hdf5/image/csv)
    detector: "H1" or "L1" (required for gps; optional hint for hdf5/csv)
    gps_time: GPS timestamp (required for gps)
    duration: seconds, one of (0.5, 1.0, 2.0, 4.0)
    """
    if input_format == "hdf5":
        if file is None:
            raise PreprocessError("hdf5 input requires `file`")
        return _from_hdf5(file, detector, duration)
    elif input_format == "gps":
        if gps_time is None or detector is None:
            raise PreprocessError("gps input requires `gps_time` and `detector`")
        return _from_gps(gps_time, detector, duration)
    elif input_format == "image":
        if file is None:
            raise PreprocessError("image input requires `file`")
        return _from_image(file)
    elif input_format == "csv":
        if file is None:
            raise PreprocessError("csv input requires `file`")
        return _from_csv(file, detector, duration)
    else:
        raise PreprocessError(
            f"Unknown input_format '{input_format}', expected one of "
            f"hdf5, gps, image, csv"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess a strain file to a Q-transform image (CLI smoke test).")
    ap.add_argument("--in", dest="inp", required=True, help="input file path")
    ap.add_argument("--format", default="hdf5", choices=["hdf5", "image", "csv"])
    ap.add_argument("--detector", default=None)
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()

    result = preprocess(args.format, file=args.inp, detector=args.detector, duration=args.duration)
    result.as_pil().save(args.out)
    print(f"Saved {args.out}. Warnings: {result.warnings or 'none'}")
