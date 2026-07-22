"""
Tests for scripts/curate_anomaly_gallery.py's segment-filtering logic.

Mocks gwosc.timeline.get_segments rather than hitting the real network.
Covers the exact bug this script had until this fix: it used to call
gwpy.segments.DataQualityFlag.query (LIGO's private, credentials-gated
segment database) with no fallback at all, which would have crashed the
script outright with a 401 Unauthorized the moment anyone ran it --
confirmed by an identical failure from the same underlying mistake
elsewhere in this codebase (preprocessing/qtransform.py's _from_gps,
fixed earlier). This script's version was actually worse: no try/except
around it at all, so it wasn't even a "warn and proceed" situation --
it would have raised before the scan loop ever started.
"""
from unittest.mock import patch

from scripts.curate_anomaly_gallery import get_available_segments, is_in_available_segment


def test_get_available_segments_returns_segments_on_success():
    with patch("scripts.curate_anomaly_gallery.get_segments") as mock_get:
        mock_get.return_value = [(1000, 2000), (3000, 4000)]
        result = get_available_segments("H1", 1000, 4000)
    assert result == [(1000, 2000), (3000, 4000)]
    mock_get.assert_called_once_with("H1_DATA", 1000, 4000)


def test_get_available_segments_casts_to_int():
    # get_segments requires integer GPS times per its own API contract;
    # float inputs (as flow through from argparse --gps-start/--gps-end)
    # must be cast, not passed through raw.
    with patch("scripts.curate_anomaly_gallery.get_segments") as mock_get:
        mock_get.return_value = []
        get_available_segments("H1", 1000.7, 4000.2)
    mock_get.assert_called_once_with("H1_DATA", 1000, 4000)


def test_get_available_segments_returns_none_on_failure_not_empty_list():
    # None ("couldn't check") must be distinguishable from [] ("checked,
    # nothing available") -- is_in_available_segment treats them
    # oppositely (None = don't filter at all, [] = filter everything out).
    with patch("scripts.curate_anomaly_gallery.get_segments", side_effect=RuntimeError("503")):
        result = get_available_segments("H1", 1000, 4000)
    assert result is None


def test_is_in_available_segment_true_when_window_fully_covered():
    segments = [(1000, 2000)]
    assert is_in_available_segment(1500, 1.0, segments) is True


def test_is_in_available_segment_false_when_window_not_covered():
    segments = [(1000, 2000)]
    assert is_in_available_segment(2500, 1.0, segments) is False


def test_is_in_available_segment_false_when_window_partially_exceeds_segment():
    segments = [(1000, 2000)]
    # window would be [1999.5, 2000.5] -- extends 0.5s past the segment end
    assert is_in_available_segment(1999.5, 1.0, segments) is False


def test_is_in_available_segment_none_means_no_filtering():
    # The fallback path: segment check failed, so scan everything and
    # let preprocess() sort out what's actually available.
    assert is_in_available_segment(999999, 1.0, None) is True
