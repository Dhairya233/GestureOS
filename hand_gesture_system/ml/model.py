"""
ml/model.py
-----------
Inference for the deployable models, in plain numpy.

Why not pickle / joblib?  Loading a pickle can run arbitrary code, and a model
file is something people download and share.  A .npz of arrays loaded with
allow_pickle=False cannot execute anything.  The price: only models whose
forward pass is a few lines of numpy can be deployed here (logistic regression
and a small MLP).  The random forest is evaluated in the experiment but not
deployed.
"""

from __future__ import annotations

import json

import numpy as np

FORMAT_VERSION = 1


class ModelError(ValueError):
    """The model file is missing, malformed or does not match this app."""


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class NumpyModel:
    """Standardise -> (logreg | relu-MLP) -> softmax."""

    def __init__(self, kind, mean, std, weights, biases, classes, meta=None) -> None:
        if kind not in ("logreg", "mlp"):
            raise ModelError(f"unknown model kind {kind!r}")
        self.kind, self.mean, self.std = kind, np.asarray(mean), np.asarray(std)
        self.weights = [np.asarray(w) for w in weights]
        self.biases = [np.asarray(b) for b in biases]
        self.classes = tuple(classes)
        self.meta = dict(meta or {})
        self._check()

    def _check(self) -> None:
        n_in = self.mean.shape[0]
        if self.std.shape != self.mean.shape or len(self.weights) != len(self.biases) or not self.weights:
            raise ModelError("inconsistent model arrays")
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != n_in or b.shape != (w.shape[1],):
                raise ModelError("weight / bias shapes do not chain")
            n_in = w.shape[1]
        if n_in != len(self.classes):
            raise ModelError("output size does not match the class list")
        if not all(np.isfinite(a).all() for a in (self.mean, self.std, *self.weights, *self.biases)):
            raise ModelError("model contains NaN or infinity")

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if x.shape[1] != self.n_features:
            raise ModelError(f"expected {self.n_features} features, got {x.shape[1]}")
        h = (x - self.mean) / np.where(self.std > 0, self.std, 1.0)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ w + b, 0.0)
        return _softmax(h @ self.weights[-1] + self.biases[-1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.classes)[self.predict_proba(x).argmax(axis=1)]

    # ── file format ───────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        meta = dict(self.meta, kind=self.kind, classes=list(self.classes),
                    n_layers=len(self.weights), format=FORMAT_VERSION)
        arrays = {"mean": self.mean, "std": self.std, "meta": np.array(json.dumps(meta))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"], arrays[f"b{i}"] = w, b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "NumpyModel":
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                if meta.get("format") != FORMAT_VERSION:
                    raise ModelError(f"unsupported model format {meta.get('format')!r}")
                n = int(meta["n_layers"])
                return cls(meta["kind"], z["mean"], z["std"],
                           [z[f"w{i}"] for i in range(n)], [z[f"b{i}"] for i in range(n)],
                           meta["classes"], meta)
        except ModelError:
            raise
        except Exception as exc:                  # missing file, bad zip, missing key...
            raise ModelError(f"cannot read model {path!r}: {type(exc).__name__}: {exc}") from exc


def from_sklearn(clf, scaler, classes=None, meta=None) -> NumpyModel:
    """Convert a fitted LogisticRegression / MLPClassifier (+ StandardScaler)."""
    classes = tuple(classes if classes is not None else clf.classes_)
    name = type(clf).__name__
    if name == "LogisticRegression":
        w, b, kind = [clf.coef_.T], [clf.intercept_], "logreg"
    elif name == "MLPClassifier":
        if clf.activation != "relu":
            raise ModelError("only relu MLPs can be exported")
        w, b, kind = list(clf.coefs_), list(clf.intercepts_), "mlp"
    else:
        raise ModelError(f"{name} cannot be exported to the numpy format")
    return NumpyModel(kind, scaler.mean_, scaler.scale_, w, b, classes, meta)
