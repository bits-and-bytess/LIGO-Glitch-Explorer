"""
End-to-end tests for the /analyze router via FastAPI's TestClient.

Uses a freshly constructed (untrained) GlitchClassifier rather than a real
trained checkpoint -- these tests exercise the API plumbing (request
parsing, response shape, file outputs, permalink retrieval), not model
quality, so an untrained model is sufficient and much faster than
training one first.
"""
import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.routers import analyze as analyze_router_module
from model.model import GRAVITY_SPY_CLASSES, GlitchClassifier
from model.ood import OODThreshold


def _wire_test_inference_service(monkeypatch, tmp_path):
    """Point the global inference_service at an untrained model + a
    results directory under tmp_path, bypassing the real weights-file
    loading path entirely (that's covered elsewhere; this is testing the
    API layer around it).
    """
    from backend.inference import InferenceService

    service = InferenceService(device="cpu")
    service.model = GlitchClassifier(pretrained=False)
    service.ood_threshold = OODThreshold(threshold=0.0)

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Both /analyze/upload (via inference_service.run) and /analyze/{id}
    # (via a live backend.inference.RESULTS_DIR lookup) resolve results
    # under this same directory, so one patch covers both endpoints.
    monkeypatch.setattr(analyze_router_module, "inference_service", service)
    monkeypatch.setattr("backend.inference.RESULTS_DIR", results_dir)
    return service, results_dir


def _make_test_png_bytes():
    img = Image.new("RGB", (224, 224), color=(40, 80, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_analyze_upload_image_returns_valid_response(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/analyze/upload",
        files={"file": ("spectrogram.png", _make_test_png_bytes(), "image/png")},
        data={"input_format": "image"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["predicted_class"] in GRAVITY_SPY_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["class_probabilities"]) == len(GRAVITY_SPY_CLASSES)
    assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-4
    assert isinstance(body["ood_score"], float)
    assert isinstance(body["ood_flagged"], bool)
    assert isinstance(body["ood_threshold"], float)
    assert body["input_format"] == "image"
    assert body["spectrogram_url"].startswith("/static/results/")
    assert body["gradcam_url"].startswith("/static/results/")
    assert "result_id" in body


def test_analyze_upload_writes_result_files_to_disk(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/analyze/upload",
        files={"file": ("spectrogram.png", _make_test_png_bytes(), "image/png")},
        data={"input_format": "image"},
    )
    result_id = response.json()["result_id"]
    result_dir = tmp_path / "results" / result_id

    assert (result_dir / "spectrogram.png").exists()
    assert (result_dir / "gradcam.png").exists()
    assert (result_dir / "meta.json").exists()


def test_analyze_permalink_retrieves_prior_result(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    first = client.post(
        "/analyze/upload",
        files={"file": ("spectrogram.png", _make_test_png_bytes(), "image/png")},
        data={"input_format": "image"},
    )
    result_id = first.json()["result_id"]

    second = client.get(f"/analyze/{result_id}")
    assert second.status_code == 200
    assert second.json()["result_id"] == result_id
    assert second.json()["predicted_class"] == first.json()["predicted_class"]


def test_analyze_unknown_permalink_returns_404(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.get("/analyze/not-a-real-result-id")
    assert response.status_code == 404


def test_analyze_rejects_bad_input_format(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/analyze/upload",
        files={"file": ("spectrogram.png", _make_test_png_bytes(), "image/png")},
        data={"input_format": "gps"},  # not valid for the /upload endpoint
    )
    assert response.status_code == 400


def test_analyze_rejects_bad_duration(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post(
        "/analyze/upload",
        files={"file": ("spectrogram.png", _make_test_png_bytes(), "image/png")},
        data={"input_format": "image", "duration": "3.7"},
    )
    assert response.status_code == 400


def test_analyze_upload_warns_on_bad_aspect_ratio(monkeypatch, tmp_path):
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    wide_img = Image.new("RGB", (600, 200), color=(10, 10, 10))
    buf = io.BytesIO()
    wide_img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/analyze/upload",
        files={"file": ("wide.png", buf, "image/png")},
        data={"input_format": "image"},
    )
    assert response.status_code == 200
    assert any("aspect ratio" in w for w in response.json()["warnings"])


def test_multiple_sequential_requests_all_succeed(monkeypatch, tmp_path):
    # Regression test for a real bug: GradCAMExplainer's torchcam hook,
    # once attached to a conv layer, intercepts EVERY subsequent forward
    # pass through that layer -- including the next request's unrelated
    # model.predict() call, which runs under @torch.no_grad() and crashes
    # if a hook from a PRIOR request's GradCAM computation is still
    # attached. A single request wouldn't catch this; it only shows up
    # across repeated requests through the same long-lived service, which
    # is exactly how the real FastAPI app behaves in production.
    _wire_test_inference_service(monkeypatch, tmp_path)
    client = TestClient(app)

    for i in range(4):
        response = client.post(
            "/analyze/upload",
            files={"file": (f"spectrogram_{i}.png", _make_test_png_bytes(), "image/png")},
            data={"input_format": "image"},
        )
        assert response.status_code == 200, f"request {i} failed: {response.text}"
        assert response.json()["predicted_class"] in GRAVITY_SPY_CLASSES
