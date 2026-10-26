"""
ml/evaluate.py
--------------
Leave-one-session-out evaluation of the learned models against the rule-based
classifier, including robustness to rotation and (simulated) distance.

Rules of the experiment, enforced by the structure of this file
---------------------------------------------------------------
* The split is by SESSION.  Consecutive frames of one recording are near
  duplicates; a random split puts almost the same frame in train and test and
  inflates accuracy.  Every row of the held-out session is unseen, and so is its
  room / lighting / posture.
* Scaling, feature statistics and model weights are fitted on the training
  sessions of each fold only.
* Nothing is tuned.  Hyper-parameters are fixed below, chosen before any data was
  looked at, so there is no test set that could have leaked into them.
* The rule-based baseline is scored on exactly the same held-out rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from . import features as F
from .classifier import RuleClassifier
from .dataset import Dataset
from .labels import CLASSES

SEED = 0
MODEL_NAMES = ("logreg", "forest", "mlp")
# conditions applied to the HELD-OUT session only: (name, rotation degrees, scale)
CONDITIONS = (("clean", 0, 1.0), ("rot30", 30, 1.0), ("rot60", 60, 1.0), ("rot90", 90, 1.0),
              ("rot180", 180, 1.0), ("scale0.5", 0, 0.5), ("scale0.75", 0, 0.75), ("scale1.5", 0, 1.5))
JITTER_PX = 1.0     # landmark noise added after scaling: a far hand has the same pixel noise
                    # on fewer pixels, which is what makes distance hard


def make_models(seed: int = SEED) -> dict:
    """Fixed hyper-parameters (see module docstring: nothing is tuned)."""
    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "mlp": MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed),
    }


def transform_points(points: np.ndarray, degrees: float, scale: float, seed: int) -> np.ndarray:
    """Rotate about the wrist, scale about the wrist, add jitter (only when degrees
    or scale change), round to integer pixels like the app does."""
    if degrees == 0 and scale == 1.0:
        return points
    p = F.rotate_points(points, degrees, about=0)
    w = p[:, :1, :]
    p = w + (p - w) * scale
    p = p + np.random.default_rng(seed).normal(0, JITTER_PX, p.shape)
    return np.round(p)


def _lms(pts: np.ndarray) -> list:
    return [(k, int(pts[k, 0]), int(pts[k, 1]), 0.0) for k in range(21)]


def rule_predictions(points: np.ndarray, hands) -> np.ndarray:
    rc = RuleClassifier()
    return np.array([rc.classify(_lms(p), h) for p, h in zip(points, hands)])


def _fit(model, X, y):
    sc = StandardScaler().fit(X)
    model.fit(sc.transform(X), y)
    return sc, model


def metrics(y_true, y_pred) -> dict:
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(CLASSES), zero_division=0)
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(f1[sup > 0].mean()) if (sup > 0).any() else 0.0,
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "confusion": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist(),
    }


def run(ds: Dataset, log=print) -> dict:
    """Full experiment.  Returns a JSON-serialisable result dict."""
    sessions = sorted(str(s) for s in set(ds.sessions))
    if len(sessions) < 2:
        raise ValueError("need at least 2 sessions for a leave-one-session-out evaluation")
    names = list(MODEL_NAMES) + ["mlp_norotate", "rules"]
    keep = {n: {c[0]: ([], []) for c in CONDITIONS} for n in names}          # (y_true, y_pred) lists
    per_session = {n: {} for n in names}

    for s in sessions:
        test = ds.sessions == s
        tr, te = ds.subset(~test), ds.subset(test)
        log(f"fold: hold out {s!r}  (train {len(tr)} rows / {len(set(tr.sessions))} sessions, test {len(te)} rows)")
        Xtr = F.feature_matrix(tr.points, tr.hands, rotate=True)
        Xtr_nr = F.feature_matrix(tr.points, tr.hands, rotate=False)
        fitted = {n: _fit(m, Xtr, tr.labels) for n, m in make_models().items()}
        fitted["mlp_norotate"] = _fit(make_models()["mlp"], Xtr_nr, tr.labels)

        for ci, (cname, deg, sc) in enumerate(CONDITIONS):
            pts = transform_points(te.points, deg, sc, seed=SEED + ci)
            Xte = F.feature_matrix(pts, te.hands, rotate=True)
            Xte_nr = F.feature_matrix(pts, te.hands, rotate=False)
            for n, (scaler, model) in fitted.items():
                pred = model.predict(scaler.transform(Xte_nr if n == "mlp_norotate" else Xte))
                keep[n][cname][0].extend(te.labels); keep[n][cname][1].extend(pred)
                if cname == "clean":
                    per_session[n][s] = float(np.mean(pred == te.labels))
            pred = rule_predictions(pts, te.hands)
            keep["rules"][cname][0].extend(te.labels); keep["rules"][cname][1].extend(pred)
            if cname == "clean":
                per_session["rules"][s] = float(np.mean(pred == te.labels))

    result = {"sessions": sessions, "n_rows": len(ds), "classes": list(CLASSES),
              "class_counts": {c: int(np.sum(ds.labels == c)) for c in CLASSES},
              "rows_per_session": {s: int(np.sum(ds.sessions == s)) for s in sessions},
              "models": {}, "conditions": [c[0] for c in CONDITIONS], "jitter_px": JITTER_PX,
              "seed": SEED}
    for n in names:
        result["models"][n] = {
            "per_session_accuracy": per_session[n],
            "clean": metrics(*keep[n]["clean"]),
            "robustness": {c: float(np.mean(np.asarray(t) == np.asarray(p)))
                           for c, (t, p) in keep[n].items()},
        }
    return result
