"""Train, evaluate and (optionally) deploy the learned gesture classifier.

    python tools/train_classifier.py data/*.csv                 # evaluate + plots + deploy MLP
    python tools/train_classifier.py data/*.csv --deploy none   # evaluate only
    python tools/train_classifier.py data/*.csv --deploy logreg

What it does
  1. Leave-one-session-out evaluation of logistic regression, random forest and a
     small MLP against the rule-based classifier (ml/evaluate.py).
  2. Plots and results.md / results.json into --out (default docs/media).
  3. Trains the chosen model on ALL sessions and writes it to
     hand_gesture_system/models/gesture_model.npz (numpy arrays, no pickle).

Input containing sessions named 'synthetic_*' is treated as a pipeline check: output
goes to out/ml_demo, plots are watermarked, and nothing is deployed.
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shown(path):
    """Path relative to the repo for display; the absolute path when it is on another drive (Windows)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path
sys.path.insert(0, os.path.join(ROOT, "hand_gesture_system"))

import numpy as np                                        # noqa: E402
import sklearn                                            # noqa: E402
from ml import dataset, evaluate, features as F, model as M, report   # noqa: E402
from ml.classifier import DEFAULT_MODEL_PATH              # noqa: E402


def deploy(ds, kind: str, path: str) -> None:
    mdl = evaluate.make_models()[kind]
    X = F.feature_matrix(ds.points, ds.hands, rotate=True)
    scaler, mdl = evaluate._fit(mdl, X, ds.labels)
    meta = {"rotate": True, "features": "coords40+tipdist10", "seed": evaluate.SEED,
            "trained_rows": int(len(ds)), "trained_sessions": len(set(ds.sessions)),
            "sklearn": sklearn.__version__, "date": datetime.date.today().isoformat()}
    npm = M.from_sklearn(mdl, scaler, meta=meta)
    # the exported numpy model must reproduce scikit-learn exactly
    ref = mdl.predict_proba(scaler.transform(X[:200]))
    got = npm.predict_proba(X[:200])
    order = [list(npm.classes).index(c) for c in mdl.classes_]
    if not np.allclose(ref, got[:, order], atol=1e-6):
        raise SystemExit("exported numpy model does not match scikit-learn; not saving")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    npm.save(path)
    print(f"saved {path} ({os.path.getsize(path) / 1024:.1f} KB, {kind})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="recorded CSV files (tools/record_gestures.py)")
    ap.add_argument("--out", default=None, help="output directory (default docs/media)")
    ap.add_argument("--deploy", choices=["mlp", "logreg", "none"], default="mlp")
    ap.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--compare", choices=["mlp", "logreg", "forest"], default="mlp",
                    help="which learned model to put beside the rules in the confusion matrix")
    a = ap.parse_args()

    ds = dataset.load(a.csv)
    sessions = sorted(set(ds.sessions))
    synthetic = any(s.startswith("synthetic_") for s in sessions)
    out = a.out or os.path.join(ROOT, "out", "ml_demo" if synthetic else os.path.join("docs", "media"))
    if not os.path.isabs(out):
        out = os.path.join(ROOT, out)
    print(f"{len(ds)} frames, {len(sessions)} sessions: {', '.join(sessions)}")
    if len(sessions) < 3:
        print("WARNING: fewer than 3 sessions. The result is not reliable; record at least 5 "
              "(different days / rooms / lighting). See docs/ML.md.")
    if synthetic:
        print("SYNTHETIC input detected: pipeline check only (nothing deployed, output in out/ml_demo).")

    res = evaluate.run(ds)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    text = report.summary_text(res, synthetic, a.compare)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(text)
    for p in report.make_plots(res, out, synthetic, a.compare):
        print("wrote", _shown(p))
    print("\n" + text)

    if a.deploy != "none" and not synthetic:
        deploy(ds, a.deploy, a.model_out)
    elif a.deploy != "none":
        print("(not deploying a model trained on synthetic data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
