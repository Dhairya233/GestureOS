"""
ml/classifier.py
----------------
Two interchangeable pose classifiers and the factory that picks one.

    rules : core.recognizer (geometry rules)        - the default
    ml    : a small numpy model trained with tools/train_classifier.py

Both expose describe(hands) -> {"Left": "Open palm", ...} and
classify(lms, label) -> a class from ml.labels.CLASSES.

Scope: the choice affects what the UI and overlay SAY the pose is.  The
controllers that move the cursor, click, scroll and so on still use their own
geometry rules.  Letting a statistical model trigger OS actions is a separate
decision (see docs/ML.md).
"""

from __future__ import annotations

import os

from core import recognizer

from . import features as F
from .labels import CLASS_TO_DISPLAY, rule_label_to_class
from .model import ModelError, NumpyModel

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "gesture_model.npz")
DEFAULT_MIN_CONFIDENCE = 0.6


class RuleClassifier:
    name = "rules"

    def classify(self, lms, label: str = "Right") -> str:
        return rule_label_to_class(recognizer.recognize(lms))

    def describe(self, hands) -> dict:
        return recognizer.describe_hands(hands)


class MLClassifier:
    name = "ml"

    def __init__(self, model: NumpyModel, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        self.model = model
        self.min_confidence = min_confidence
        self.rotate = bool(model.meta.get("rotate", True))
        if model.n_features != F.N_FEATURES:
            raise ModelError(f"model expects {model.n_features} features, app produces {F.N_FEATURES}")

    def classify_with_confidence(self, lms, label: str = "Right"):
        p = self.model.predict_proba(F.features_for_hand(lms, label, self.rotate)[None])[0]
        i = int(p.argmax())
        cls = self.model.classes[i]
        return (cls if p[i] >= self.min_confidence else "none"), float(p[i])

    def classify(self, lms, label: str = "Right") -> str:
        return self.classify_with_confidence(lms, label)[0]

    def describe(self, hands) -> dict:
        return {h["label"]: CLASS_TO_DISPLAY.get(self.classify(h["landmarks"], h["label"]), "Unknown")
                for h in hands or []}


def make_classifier(name: str, model_path: str = DEFAULT_MODEL_PATH, log=None):
    """Return (classifier, note).  `note` is '' normally, or a sentence saying why
    the requested classifier could not be used (the caller shows it once).  The
    app never fails to start because a model is missing or broken."""
    if name != "ml":
        return RuleClassifier(), ""
    try:
        return MLClassifier(NumpyModel.load(model_path)), ""
    except ModelError as exc:
        note = f"ML classifier unavailable, using rules instead: {exc}"
        if log:
            log(note)
        return RuleClassifier(), note
