"""
Tests for the HDF5 preprocessing path (Format 1: "the most common format
scientists will use" per spec).

Fixtures replicate the REAL GWOSC HDF5 structure, cross-checked against
gwpy's actual reader source (gwpy.timeseries.io.losc.read_gwosc_hdf5):
    /strain/Strain  -- dataset with Xspacing, Xstart, Xunits, Yunits attrs
    /meta/Detector   -- scalar dataset, e.g. b'H1'

These tests require gwpy (and therefore run slower than the rest of the
suite -- real Q-transform + whitening computation, not mocked).
"""
import h5py
import numpy as np
import pytest

from preprocessing.qtransform import (
    MIN_PAD_FOR_WHITENING_SECONDS,
    PreprocessError,
    _compute_qtransform_image,
    preprocess,
)

SAMPLE_RATE = 4096
GPS_START = 1369062010.0


def _make_gwosc_h5(path, detector="H1", sample_rate=SAMPLE_RATE, duration_s=8.0,
                    gps_start=GPS_START, nan_frac_of_output_window=0.0, output_duration=1.0):
    """Build a synthetic file matching the real GWOSC HDF5 structure.
    If nan_frac_of_output_window > 0, places a NaN gap sized as that exact
    fraction of the (file-centered) output window -- calibrated against
    the window, not the whole file, so overlap fraction is precise.
    """
    n = int(sample_rate * duration_s)
    rng = np.random.default_rng(42)
    strain = rng.normal(0, 1e-21, size=n).astype("float64")

    if nan_frac_of_output_window > 0:
        center_sample = n // 2
        gap_samples = int(sample_rate * output_duration * nan_frac_of_output_window)
        start_idx = center_sample - gap_samples // 2
        strain[start_idx:start_idx + gap_samples] = np.nan

    with h5py.File(path, "w") as f:
        ds = f.create_group("strain").create_dataset("Strain", data=strain)
        ds.attrs["Xspacing"] = 1.0 / sample_rate
        ds.attrs["Xstart"] = float(gps_start)
        ds.attrs["Xunits"] = "second"
        ds.attrs["Yunits"] = "strain"  # gwpy's reader requires this; easy to miss
        f.create_group("meta").create_dataset("Detector", data=detector.encode())


def test_clean_file_produces_valid_image(tmp_path):
    path = tmp_path / "clean.h5"
    _make_gwosc_h5(path)
    result = preprocess("hdf5", file=str(path), duration=1.0)
    assert result.image.shape == (224, 224, 3)
    assert result.sample_rate == SAMPLE_RATE
    assert result.detector == "H1"


def test_detector_and_sample_rate_read_from_file_metadata(tmp_path):
    path = tmp_path / "l1.h5"
    _make_gwosc_h5(path, detector="L1", sample_rate=16384)
    result = preprocess("hdf5", file=str(path), duration=1.0)
    assert result.detector == "L1"
    assert result.sample_rate == 16384


def test_partial_nan_gap_in_output_window_warns_but_succeeds(tmp_path):
    path = tmp_path / "partial_gap.h5"
    _make_gwosc_h5(path, nan_frac_of_output_window=0.2)
    result = preprocess("hdf5", file=str(path), duration=1.0)
    assert result.image.shape == (224, 224, 3)
    assert any("missing/invalid strain data" in w for w in result.warnings)


def test_mostly_nan_output_window_raises(tmp_path):
    path = tmp_path / "mostly_nan.h5"
    _make_gwosc_h5(path, nan_frac_of_output_window=0.7)
    with pytest.raises(PreprocessError, match="no valid strain data"):
        preprocess("hdf5", file=str(path), duration=1.0)


def test_missing_strain_dataset_raises_clear_error(tmp_path):
    path = tmp_path / "not_gwosc.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("some_other_data", data=[1, 2, 3])
    with pytest.raises(PreprocessError, match="strain/Strain"):
        preprocess("hdf5", file=str(path), duration=1.0)


def test_unrecognized_detector_warns_but_proceeds(tmp_path):
    path = tmp_path / "unknown_detector.h5"
    _make_gwosc_h5(path, detector="V1")  # Virgo -- explicitly out of scope per spec
    result = preprocess("hdf5", file=str(path), duration=1.0)
    assert any("not recognized/supported" in w for w in result.warnings)


class TestQtransformPaddingAndWhitening:
    """Direct tests of _compute_qtransform_image's padding/whitening logic,
    which is shared by all three gwpy-backed input paths (hdf5/gps/csv).
    """

    def _make_ts(self, duration_s=8.0, sample_rate=SAMPLE_RATE):
        from gwpy.timeseries import TimeSeries
        rng = np.random.default_rng(0)
        n = int(sample_rate * duration_s)
        return TimeSeries(rng.normal(0, 1e-21, n), sample_rate=sample_rate, t0=GPS_START)

    def test_ample_padding_whitens_without_warning(self):
        # A file with generous padding around the output window shouldn't
        # trigger either the hard whiten=False path or the soft padding warning.
        ts = self._make_ts(duration_s=20.0)  # 9.5s padding each side of a 1s window
        warnings = []
        image = _compute_qtransform_image(ts, 1.0, warnings)
        assert image.shape == (224, 224, 3)
        assert not any("padding" in w for w in warnings)

    def test_thin_padding_warns_softly_but_still_whitens(self):
        ts = self._make_ts(duration_s=8.0)  # 3.5s each side: below target, above hard minimum
        warnings = []
        _compute_qtransform_image(ts, 1.0, warnings)
        assert any("less stable than usual" in w for w in warnings)

    def test_insufficient_padding_disables_whitening_without_crashing(self):
        ts = self._make_ts(duration_s=1.2)  # 0.1s each side: below MIN_PAD_FOR_WHITENING_SECONDS
        warnings = []
        image = _compute_qtransform_image(ts, 1.0, warnings)
        assert image.shape == (224, 224, 3)
        assert any("Not enough surrounding data to whiten" in w for w in warnings)

    def test_window_entirely_outside_available_data_raises(self):
        ts = self._make_ts(duration_s=8.0)
        with pytest.raises(PreprocessError, match="doesn't overlap"):
            _compute_qtransform_image(ts, 1.0, [], center_gps=GPS_START + 10_000)

    def test_min_pad_constant_is_positive_and_below_target(self):
        # Sanity check on the constants themselves -- if someone edits
        # these later, this keeps the two from being silently inverted.
        from preprocessing.qtransform import TARGET_WHITEN_PAD_SECONDS
        assert 0 < MIN_PAD_FOR_WHITENING_SECONDS < TARGET_WHITEN_PAD_SECONDS
