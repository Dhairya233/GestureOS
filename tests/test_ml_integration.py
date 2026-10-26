"""The classifier setting through the real engine, and the Settings control."""
import numpy as np

from test_engine_smoke import engine, frame        # noqa: F401  (fixture reuse)
from conftest import make_hand
from core.status import STATUS, TIMELINE
from ml.classifier import MLClassifier, RuleClassifier
from ml.model import NumpyModel
from ml import features as F
from ml.labels import CLASSES


def test_default_config_selects_rules():
    import config_manager
    assert config_manager.DEFAULTS["classifier"] == "rules"


def test_engine_falls_back_to_rules_when_model_missing(engine, tmp_path, monkeypatch):  # noqa: F811
    import ml.classifier as mc
    eng, _ = engine
    monkeypatch.setattr(mc, "DEFAULT_MODEL_PATH", str(tmp_path / "absent.npz"))
    import main
    monkeypatch.setattr(main, "make_classifier", lambda name, log=None: mc.make_classifier(name, str(tmp_path / "absent.npz"), log))
    TIMELINE.clear()
    eng._select_classifier("ml")
    assert isinstance(eng._classifier, RuleClassifier)
    st = STATUS.snapshot()
    assert st.classifier == "rules" and "using rules instead" in st.classifier_note
    eng._tracker.hands = [make_hand("Right", pose="open")]
    eng._process_frame(frame())                               # still works, shows rule names
    assert STATUS.snapshot().gestures == {"Right": "Open palm"}
    assert any("fallback" in e.text for e in TIMELINE.since(0))


def test_engine_uses_ml_classifier_for_display_only(engine):  # noqa: F811
    eng, pg = engine
    b = np.zeros(len(CLASSES)); b[CLASSES.index("fist")] = 10.0
    model = NumpyModel("logreg", np.zeros(F.N_FEATURES), np.ones(F.N_FEATURES),
                       [np.zeros((F.N_FEATURES, len(CLASSES)))], [b], CLASSES)
    eng._classifier, eng._cls_requested = MLClassifier(model), "ml"
    eng.arming.set_armed(True)
    eng._frame_count = 1          # frame 0 would reload the config (classifier: rules) and replace the fake
    # an open palm in the scroll band: the SCROLL CONTROLLER (rules) must still scroll,
    # even though the display classifier (this fake model) calls everything a fist
    eng._tracker.hands = [make_hand("Right", pose="open", origin=(300, 440))]
    eng._process_frame(frame())
    assert STATUS.snapshot().gestures == {"Right": "Fist"} and STATUS.snapshot().classifier == "ml"
    assert any(c[0] == "scroll" for c in pg.calls)


def test_classifier_exception_does_not_stop_the_frame(engine):  # noqa: F811
    eng, _ = engine
    class Broken:
        name = "ml"
        def describe(self, hands): raise RuntimeError("boom")
    eng._classifier = Broken()
    eng._tracker.hands = [make_hand("Right", pose="open")]
    eng._process_frame(frame())
    assert eng.latest_frame is not None


def test_engine_records_stage_timings(engine):  # noqa: F811
    eng, _ = engine
    eng._tracker.hands = [make_hand("Right", pose="open")]
    for _ in range(3):
        eng._process_frame(frame())
    s = eng.timings.summary()
    assert {"mediapipe", "classify", "controllers", "overlay", "frame"} <= set(s)
    assert all(v["mean_ms"] >= 0 and v["p95_ms"] >= v["mean_ms"] * 0.0 for v in s.values())
    assert s["frame"]["mean_ms"] >= s["controllers"]["mean_ms"]
