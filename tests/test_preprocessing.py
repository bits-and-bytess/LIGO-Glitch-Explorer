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
