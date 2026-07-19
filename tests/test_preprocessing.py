import io

import pytest
from PIL import Image

from preprocessing.qtransform import IMAGE_SIZE, PreprocessError, preprocess


def _png_bytes(width, height, color=(50, 100, 150)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_image_format_resizes_to_expected_shape():
    result = preprocess("image", file=_png_bytes(300, 300))
    assert result.image.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    assert result.source_format == "image"
    assert result.warnings == []


def test_image_format_warns_on_bad_aspect_ratio():
    result = preprocess("image", file=_png_bytes(600, 200))
    assert any("aspect ratio" in w for w in result.warnings)


def test_image_format_warns_on_low_resolution():
    result = preprocess("image", file=_png_bytes(32, 32))
    assert any("resolution" in w for w in result.warnings)


def test_unknown_input_format_raises():
    with pytest.raises(PreprocessError):
        preprocess("carrier_pigeon", file=_png_bytes(224, 224))


def test_image_missing_file_raises():
    with pytest.raises(PreprocessError):
        preprocess("image", file=None)


def test_csv_missing_columns_raises():
    csv_bytes = io.BytesIO(b"foo,bar\n1,2\n3,4\n")
    with pytest.raises(PreprocessError):
        preprocess("csv", file=csv_bytes)


def test_csv_nonmonotonic_time_raises():
    csv_bytes = io.BytesIO(b"time,strain\n0.0,0.1\n0.5,0.2\n0.1,0.3\n")
    with pytest.raises(PreprocessError):
        preprocess("csv", file=csv_bytes)


def test_csv_with_sufficient_samples_produces_valid_image():
    # Enough samples (8s at 4096Hz) for a real Q-transform, not just the
    # validation-error branches tested above.
    import numpy as np
    import pandas as pd

    sample_rate = 4096
    n = sample_rate * 8
    t = np.arange(n) / sample_rate + 1369062010.0
    rng = np.random.default_rng(0)
    strain = rng.normal(0, 1e-21, n)

    buf = io.StringIO()
    pd.DataFrame({"time": t, "strain": strain}).to_csv(buf, index=False)
    csv_bytes = io.BytesIO(buf.getvalue().encode())

    result = preprocess("csv", file=csv_bytes, duration=1.0)
    assert result.image.shape == (224, 224, 3)
    assert abs(result.sample_rate - sample_rate) < 1.0


def test_gps_nan_time_raises_clear_error_instead_of_hitting_gwosc():
    # Real bug: a blank/unparseable GPS time field on the frontend sends
    # NaN (Python's float("nan") parses successfully, so nothing upstream
    # rejected it), which used to propagate all the way to a GWOSC fetch
    # call and fail with a confusing low-level error ("cannot convert
    # float NaN to integer") instead of a clear, actionable message.
    from preprocessing.qtransform import _from_gps

    with pytest.raises(PreprocessError, match="finite number"):
        _from_gps(float("nan"), "H1", 1.0)


def test_gps_infinite_time_raises():
    from preprocessing.qtransform import _from_gps

    with pytest.raises(PreprocessError, match="finite number"):
        _from_gps(float("inf"), "H1", 1.0)


def test_gps_negative_time_raises():
    from preprocessing.qtransform import _from_gps

    with pytest.raises(PreprocessError, match="non-negative"):
        _from_gps(-100.0, "H1", 1.0)
