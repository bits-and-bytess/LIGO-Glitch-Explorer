import torch

from model.model import GRAVITY_SPY_CLASSES, NUM_CLASSES, GlitchClassifier
from model.ood import OODThreshold, energy_score


def test_num_classes_matches_class_list():
    assert NUM_CLASSES == len(GRAVITY_SPY_CLASSES)


def test_forward_pass_shape():
    model = GlitchClassifier(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, NUM_CLASSES)


def test_embedding_dim_matches_classifier_input():
    model = GlitchClassifier(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    emb = model.embedding(x)
    assert emb.shape == (1, model.embed_dim)
    assert model.classifier.in_features == model.embed_dim


def test_predict_returns_valid_probability_distribution():
    model = GlitchClassifier(pretrained=False)
    x = torch.randn(3, 3, 224, 224)
    pred, probs, logits, emb = model.predict(x)
    assert pred.shape == (3,)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-4)
    assert (pred == probs.argmax(dim=1)).all()


def test_energy_score_lower_for_confident_logits():
    # A sharply-peaked logit vector should get a lower (more in-distribution)
    # energy score than a flat/uncertain one.
    confident = torch.tensor([[10.0] + [0.0] * (NUM_CLASSES - 1)])
    flat = torch.zeros((1, NUM_CLASSES))
    assert energy_score(confident).item() < energy_score(flat).item()


def test_ood_threshold_fit_and_is_ood():
    torch.manual_seed(0)
    val_logits = torch.randn(200, NUM_CLASSES) * 2  # pretend in-distribution val logits
    threshold = OODThreshold.fit(val_logits, percentile=95.0)

    # A logit vector far more extreme than anything in val_logits should
    # very likely score as OOD-eligible on the low-energy side isn't
    # guaranteed, but a clearly flat/uncertain vector near the threshold
    # boundary should at least produce a valid boolean + float without error.
    is_ood, score = threshold.is_ood(val_logits[:1])
    assert isinstance(is_ood, bool)
    assert isinstance(score, float)


def test_ood_interpretation_is_nonempty_string():
    threshold = OODThreshold(threshold=1.0)
    for score in (0.0, 1.0, 5.0):
        msg = threshold.interpretation(score)
        assert isinstance(msg, str) and len(msg) > 0
