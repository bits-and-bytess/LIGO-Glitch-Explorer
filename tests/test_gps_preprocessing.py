"""
Tests for the GPS+detector preprocessing path (_from_gps).

Mocks gwosc.timeline.get_segments and TimeSeries.fetch_open_data rather
than hitting the real network -- this sandbox can't reach gwosc.org, and
even where it can, tests shouldn't depend on live network availability.
The mocked fetch_open_data returns a real synthetic TimeSeries so the
rest of the pipeline (Q-transform, rasterization) runs for real on it,
not just the segment-check logic in isolation.

This specifically covers a real bug: the original science-segment check
used gwpy.segments.DataQualityFlag.query, which hits LIGO's private,
credentials-required segment database and failed with a live 401
Unauthorized for an actual public user. Fixed to use gwosc.timeline's
public API instead.
"""
from unittest.mock import patch

import numpy as np
import pytest
from gwpy.timeseries import TimeSeries

from preprocessing.qtransform import PreprocessError, _from_gps

GPS_START = 1126259462.0  # GW150914, used throughout as a realistic example time


def _fake_timeseries(center, duration_s=20.0, sample_rate=4096):
    rng = np.random.default_rng(0)
    n = int(sample_rate * duration_s)
    return TimeSeries(rng.normal(0, 1e-21, n), sample_rate=sample_rate, t0=center - duration_s / 2)


def test_invalid_detector_raises():
    with pytest.raises(PreprocessError, match="detector must be one of"):
        _from_gps(GPS_START, "V1", 1.0)


def test_window_fully_covered_by_public_segment_produces_no_warning():
    with patch("gwosc.timeline.get_segments") as mock_segments, \
         patch.object(TimeSeries, "fetch_open_data", return_value=_fake_timeseries(GPS_START)):
        # A single segment comfortably covering the whole padded window
        mock_segments.return_value = [(int(GPS_START) - 1000, int(GPS_START) + 1000)]
        result = _from_gps(GPS_START, "H1", 1.0)

    assert result.image.shape == (224, 224, 3)
    assert not any("science-mode" in w for w in result.warnings)


def test_window_not_covered_by_any_public_segment_warns():
    with patch("gwosc.timeline.get_segments") as mock_segments, \
         patch.object(TimeSeries, "fetch_open_data", return_value=_fake_timeseries(GPS_START)):
        mock_segments.return_value = []  # no public data-available segments at all
        result = _from_gps(GPS_START, "H1", 1.0)

    assert any("does not fall fully within" in w for w in result.warnings)


def test_segment_check_failure_warns_but_does_not_block_analysis():
    # The actual real-world case: the segment check itself fails (was a
    # 401 Unauthorized before the fix; could be any network hiccup after
    # it), but the analysis should still proceed using fetch_open_data
    # directly rather than hard-failing on the pre-check alone.
    with patch("gwosc.timeline.get_segments", side_effect=RuntimeError("503 Service Unavailable")), \
         patch.object(TimeSeries, "fetch_open_data", return_value=_fake_timeseries(GPS_START)):
        result = _from_gps(GPS_START, "H1", 1.0)

    assert result.image.shape == (224, 224, 3)
    assert any("Could not verify science-mode segment" in w for w in result.warnings)


def test_fetch_failure_raises_clear_preprocess_error():
    with patch("gwosc.timeline.get_segments", return_value=[(int(GPS_START) - 1000, int(GPS_START) + 1000)]), \
         patch.object(TimeSeries, "fetch_open_data", side_effect=RuntimeError("no data at this time")):
        with pytest.raises(PreprocessError, match="Failed to fetch GWOSC open data"):
            _from_gps(GPS_START, "H1", 1.0)
