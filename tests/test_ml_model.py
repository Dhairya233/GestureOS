"""The deployable numpy model: matches scikit-learn, round-trips, and refuses bad files."""
import io
import zipfile

import numpy as np
import pytest

from ml import features as F
from ml.classifier import MLClassifier, RuleClassifier, make_classifier
from ml.labels import CLASSES
from ml.model import ModelError, NumpyModel, from_sklearn
from conftest import make_hand

sk = pytest.importorskip("sklearn")
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _fitted(kind):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, F.N_FEATURES))
    y = np.array(CLASSES)[rng.integers(0, len(CLASSES), 300)]
    sc = StandardScaler().fit(X)
    m = (LogisticRegression(max_iter=500, random_state=0) if kind == "logreg"
         else MLPClassifier(hidden_layer_sizes=(16,), max_iter=300, random_state=0)).fit(sc.transform(X), y)
    return X, sc, m


@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_numpy_inference_equals_sklearn(kind):
    X, sc, m = _fitted(kind)
    npm = from_sklearn(m, sc)
    ref = m.predict_proba(sc.transform(X))
    np.testing.assert_allclose(npm.predict_proba(X), ref, atol=1e-9)
    assert (npm.predict(X) == m.predict(sc.transform(X))).all()


@pytest.mark.parametrize("kind", ["logreg", "mlp"])
def test_save_load_roundtrip_and_small(tmp_path, kind):
    X, sc, m = _fitted(kind)
    npm = from_sklearn(m, sc, meta={"rotate": True})
    p = str(tmp_path / "m.npz"); npm.save(p)
    back = NumpyModel.load(p)
    np.testing.assert_allclose(back.predict_proba(X[:20]), npm.predict_proba(X[:20]))
    assert back.meta["rotate"] is True
    import os
    assert os.path.getsize(p) < 1_000_000


def test_forest_cannot_be_exported():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    X = np.random.default_rng(0).normal(size=(30, 4)); y = np.array(["a", "b"] * 15)
    with pytest.raises(ModelError):
        from_sklearn(RandomForestClassifier(n_estimators=3).fit(X, y), StandardScaler().fit(X))


def _write_npz(path, **arrays):
    np.savez(path, **arrays)


def test_load_refuses_missing_garbage_and_pickled_files(tmp_path):
    with pytest.raises(ModelError):
        NumpyModel.load(str(tmp_path / "nope.npz"))
    g = tmp_path / "g.npz"; g.write_bytes(b"not a zip")
    with pytest.raises(ModelError):
        NumpyModel.load(str(g))
    # an npz whose 'meta' entry is a pickled python object must NOT be executed
    class Boom:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwned > " + str(tmp_path / "pwned.txt"),))
    bad = tmp_path / "bad.npz"
    buf = io.BytesIO(); np.save(buf, np.array([Boom()], dtype=object), allow_pickle=True)
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("meta.npy", buf.getvalue())
    with pytest.raises(ModelError):
        NumpyModel.load(str(bad))
    assert not (tmp_path / "pwned.txt").exists()


def test_load_refuses_inconsistent_or_non_finite_models(tmp_path):
    X, sc, m = _fitted("logreg")
    good = from_sklearn(m, sc)
    w = good.weights[0].copy(); w[0, 0] = np.nan
    with pytest.raises(ModelError):
        NumpyModel("logreg", good.mean, good.std, [w], good.biases, good.classes)
    with pytest.raises(ModelError):
        NumpyModel("logreg", good.mean, good.std, [good.weights[0][:, :3]], [good.biases[0][:3]], good.classes)
    with pytest.raises(ModelError):
        NumpyModel("forest", good.mean, good.std, good.weights, good.biases, good.classes)
    with pytest.raises(ModelError):
        good.predict_proba(np.zeros((1, 7)))


def test_wrong_feature_count_is_rejected_by_the_classifier():
    small = NumpyModel("logreg", np.zeros(5), np.ones(5), [np.zeros((5, 2))], [np.zeros(2)], ("a", "b"))
    with pytest.raises(ModelError):
        MLClassifier(small)


def test_missing_model_falls_back_to_rules_and_says_why(tmp_path):
    logged = []
    clf, note = make_classifier("ml", str(tmp_path / "absent.npz"), log=logged.append)
    assert isinstance(clf, RuleClassifier) and "using rules instead" in note and logged == [note]


def test_rules_is_the_default_and_unknown_names_mean_rules():
    for name in ("rules", "", "banana"):
        clf, note = make_classifier(name)
        assert isinstance(clf, RuleClassifier) and note == ""


def test_low_confidence_becomes_none(tmp_path):
    # 2-class-ish model whose softmax is ~0.5 for everything: below the threshold
    w = np.zeros((F.N_FEATURES, len(CLASSES)))
    m = NumpyModel("logreg", np.zeros(F.N_FEATURES), np.ones(F.N_FEATURES), [w], [np.zeros(len(CLASSES))], CLASSES)
    clf = MLClassifier(m, min_confidence=0.6)
    hand = make_hand("Right", pose="open")
    assert clf.classify(hand["landmarks"], "Right") == "none"
    assert clf.describe([hand]) == {"Right": "Unknown"}


def test_ml_describe_uses_the_recogniser_wording():
    b = np.zeros(len(CLASSES)); b[CLASSES.index("fist")] = 10.0
    m = NumpyModel("logreg", np.zeros(F.N_FEATURES), np.ones(F.N_FEATURES),
                   [np.zeros((F.N_FEATURES, len(CLASSES)))], [b], CLASSES)
    assert MLClassifier(m).describe([make_hand("Left", pose="open")]) == {"Left": "Fist"}
    assert MLClassifier(m).describe(None) == {}
