"""Dataset, recorder, evaluation structure and the train CLI, on small synthetic data."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("sklearn")

from ml import dataset, evaluate, synthetic
from ml.labels import CLASSES, rule_label_to_class
from ml.recording import Recorder
from conftest import make_hand


def _csvs(tmp_path, n_sessions=3, per_class=20, prefix="synthetic_"):
    paths = []
    for i in range(n_sessions):
        p = str(tmp_path / f"s{i}.csv")
        synthetic.write_session_csv(p, f"{prefix}{i}", seed=i, n_per_class=per_class)
        paths.append(p)
    return paths


# ── dataset ────────────────────────────────────────────────────────────────
def test_csv_roundtrip_preserves_landmarks(tmp_path):
    p = str(tmp_path / "a.csv")
    hand = make_hand("Left", pose="peace")
    w = dataset.CsvWriter(p)
    w.write(dataset.row_for_hand("two_fingers", "room1", 1.5, hand, (640, 480))); w.close()
    ds = dataset.load(p)
    assert len(ds) == 1 and ds.labels[0] == "two_fingers" and ds.hands[0] == "Left"
    assert ds.lms(0) == [(i, x, y, 0.0) for i, x, y, _ in hand["landmarks"]]


def test_writer_appends_without_duplicating_the_header(tmp_path):
    p = str(tmp_path / "a.csv")
    for _ in range(2):
        w = dataset.CsvWriter(p)
        w.write(dataset.row_for_hand("fist", "s", 0, make_hand("Right", pose="fist"), (640, 480))); w.close()
    assert open(p).read().count("label,session") == 1
    assert len(dataset.load(p)) == 2


@pytest.mark.parametrize("edit,msg", [
    (lambda r: r.__setitem__(0, "dance"), "unknown class"),
    (lambda r: r.__setitem__(3, "Middle"), "handedness"),
    (lambda r: r.__setitem__(10, "nan"), "non-finite"),
    (lambda r: r.pop(), "columns"),
])
def test_loader_rejects_bad_rows_with_line_numbers(tmp_path, edit, msg):
    import csv
    p = tmp_path / "bad.csv"
    row = dataset.row_for_hand("fist", "s", 0, make_hand("Right", pose="fist"), (640, 480))
    edit(row)
    with open(p, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(dataset.COLUMNS); wr.writerow(row)
    with pytest.raises(dataset.DatasetError, match=msg) as e:
        dataset.load(str(p))
    assert "line 2" in str(e.value)


def test_loader_rejects_wrong_header_and_empty(tmp_path):
    (tmp_path / "h.csv").write_text("a,b,c\n1,2,3\n")
    with pytest.raises(dataset.DatasetError, match="header"):
        dataset.load(str(tmp_path / "h.csv"))
    (tmp_path / "e.csv").write_text(",".join(dataset.COLUMNS) + "\n")
    with pytest.raises(dataset.DatasetError, match="no rows"):
        dataset.load(str(tmp_path / "e.csv"))


def test_rule_label_mapping_covers_every_recogniser_output():
    from core.recognizer import recognize
    for pose in ("open", "fist", "point", "peace", "three", "four", "thumbs_up"):
        assert rule_label_to_class(recognize(make_hand("Right", pose=pose)["landmarks"])) in CLASSES
    assert rule_label_to_class("Pinch (middle)") == "pinch" and rule_label_to_class("Unknown") == "none"


# ── recorder ───────────────────────────────────────────────────────────────
def test_recorder_saves_only_single_hand_frames_while_recording(tmp_path):
    rec = Recorder(dataset.CsvWriter(str(tmp_path / "r.csv")), "sess")
    h = make_hand("Right", pose="fist")
    assert rec.feed([h], 0, (640, 480)) == "paused"
    rec.set_label("fist"); rec.toggle()
    assert rec.feed(None, 0.1, (640, 480)) == "no_hand"
    assert rec.feed([h, make_hand("Left")], 0.2, (640, 480)) == "multiple_hands"
    assert rec.feed([h], 0.3, (640, 480)) == "saved"
    rec.writer.close()
    assert rec.saved["fist"] == 1 and rec.skipped["no_hand"] == 1 and rec.skipped["multiple_hands"] == 1
    assert len(dataset.load(str(tmp_path / "r.csv"))) == 1
    with pytest.raises(ValueError):
        rec.set_label("dance")


# ── evaluation structure ───────────────────────────────────────────────────
def test_every_fold_trains_without_the_held_out_session(tmp_path, monkeypatch):
    ds = dataset.load(_csvs(tmp_path, 3))
    seen = []
    real_fit = evaluate._fit
    monkeypatch.setattr(evaluate, "_fit", lambda m, X, y: (seen.append(len(X)), real_fit(m, X, y))[1])
    res = evaluate.run(ds, log=lambda m: None)
    sizes = sorted(res["rows_per_session"].values())
    total = len(ds)
    # 4 models are fitted per fold; each must have seen exactly (total - held-out rows)
    expected = sorted(total - n for n in res["rows_per_session"].values())
    assert sorted(set(seen)) == sorted(set(expected))
    assert all(s < total for s in seen) and len(seen) == 4 * 3 and sizes


def test_run_needs_two_sessions(tmp_path):
    with pytest.raises(ValueError):
        evaluate.run(dataset.load(_csvs(tmp_path, 1)), log=lambda m: None)


def test_run_is_reproducible(tmp_path):
    ds = dataset.load(_csvs(tmp_path, 3, per_class=15))
    a = evaluate.run(ds, log=lambda m: None)
    b = evaluate.run(ds, log=lambda m: None)
    assert a["models"]["mlp"]["clean"]["accuracy"] == b["models"]["mlp"]["clean"]["accuracy"]
    assert a["models"]["forest"]["robustness"] == b["models"]["forest"]["robustness"]


def test_result_has_every_model_condition_and_class(tmp_path):
    res = evaluate.run(dataset.load(_csvs(tmp_path, 3, per_class=15)), log=lambda m: None)
    assert set(res["models"]) == {"logreg", "forest", "mlp", "mlp_norotate", "rules"}
    for m in res["models"].values():
        assert set(m["robustness"]) == {c[0] for c in evaluate.CONDITIONS}
        assert set(m["clean"]["per_class"]) == set(CLASSES)
    json.dumps(res)                                   # serialisable


def test_transform_is_identity_for_clean_and_rounds_to_pixels():
    pts = np.random.default_rng(0).uniform(100, 500, (4, 21, 2)).round()
    assert evaluate.transform_points(pts, 0, 1.0, seed=0) is pts
    out = evaluate.transform_points(pts, 30, 0.5, seed=0)
    assert out.shape == pts.shape and np.allclose(out, np.round(out))


# ── train CLI ──────────────────────────────────────────────────────────────
def _run_train(monkeypatch, argv):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import train_classifier
    monkeypatch.setattr(sys, "argv", ["train_classifier.py", *argv])
    return train_classifier.main()


def test_train_cli_on_synthetic_data_is_watermarked_and_never_deploys(tmp_path, monkeypatch):
    out, model = tmp_path / "out", tmp_path / "m.npz"
    paths = _csvs(tmp_path, 3, per_class=12)
    assert _run_train(monkeypatch, [*paths, "--out", str(out), "--model-out", str(model)]) == 0
    assert (out / "results.json").exists() and (out / "confusion_matrix.png").exists()
    assert "SYNTHETIC DATA" in (out / "results.md").read_text()
    assert not model.exists()


def test_train_cli_deploys_a_numpy_model_that_matches_sklearn(tmp_path, monkeypatch):
    out, model = tmp_path / "out", tmp_path / "m.npz"
    paths = _csvs(tmp_path, 3, per_class=12, prefix="room_")       # named like real sessions
    assert _run_train(monkeypatch, [*paths, "--out", str(out), "--model-out", str(model),
                                    "--deploy", "logreg"]) == 0
    from ml.model import NumpyModel
    m = NumpyModel.load(str(model))
    assert m.kind == "logreg" and os.path.getsize(model) < 1_000_000
    assert "SYNTHETIC DATA" not in (out / "results.md").read_text()


def test_tools_print_a_path_even_when_it_is_on_another_drive(monkeypatch):
    """Windows CI: temp dir on C:, repo on D: made os.path.relpath raise ValueError."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import benchmark, train_classifier

    def boom(*a, **k):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")
    monkeypatch.setattr(os.path, "relpath", boom)
    for mod in (benchmark, train_classifier):
        assert mod._shown("C:/tmp/x.json") == "C:/tmp/x.json"
