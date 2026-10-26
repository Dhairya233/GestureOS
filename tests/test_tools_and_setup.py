"""Config robustness, model download/verification, run.py checks, pinned requirements."""
import hashlib
import importlib.util
import io
import json
import os
import re
import urllib.error

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── config ─────────────────────────────────────────────────────────────────
@pytest.fixture
def cfgmod(tmp_path, monkeypatch):
    import config_manager as cm
    monkeypatch.setattr(cm, "CONFIG_PATH", str(tmp_path / "config.json"))
    return cm


def test_first_run_creates_config_from_defaults(cfgmod):
    cfg = cfgmod.load_config()
    assert os.path.exists(cfgmod.CONFIG_PATH)
    assert cfg == cfgmod.DEFAULTS


def test_corrupt_config_is_restored(cfgmod):
    open(cfgmod.CONFIG_PATH, "w").write("{not json")
    cfg = cfgmod.load_config()
    assert cfg == cfgmod.DEFAULTS
    assert json.load(open(cfgmod.CONFIG_PATH)) == cfgmod.DEFAULTS


def test_partial_config_is_backfilled_and_nested_dicts_merge(cfgmod):
    json.dump({"scroll_speed": 0.9, "app_paths": {"2_fingers": "notepad"}}, open(cfgmod.CONFIG_PATH, "w"))
    cfg = cfgmod.load_config()
    assert cfg["scroll_speed"] == 0.9
    assert cfg["app_paths"]["2_fingers"] == "notepad"
    assert cfg["app_paths"]["3_fingers"] == cfgmod.DEFAULTS["app_paths"]["3_fingers"]
    assert cfg["enable_mouse"] is True


def test_save_is_atomic_and_leaves_no_temp_files(cfgmod, tmp_path):
    cfg = cfgmod.load_config(); cfg["mouse_smoothing"] = 9
    cfgmod.save_config(cfg)
    assert json.load(open(cfgmod.CONFIG_PATH))["mouse_smoothing"] == 9
    assert [p.name for p in tmp_path.iterdir()] == ["config.json"]


def test_defaults_file_matches_documented_keys(cfgmod):
    d = cfgmod.DEFAULTS
    for k in ("enable_mouse", "enable_kill", "kill_frames", "swap_handedness", "app_paths", "mouse_smoothing"):
        assert k in d
    assert not (set(d) & set(cfgmod.RETIRED_KEYS))


# ── model download / checksum ──────────────────────────────────────────────
@pytest.fixture
def dm(monkeypatch):
    mod = load("tools/download_model.py", "download_model")
    return mod


def fake_urlopen(payload):
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return lambda url, timeout=30: R(payload)


def test_valid_download_is_accepted(dm, monkeypatch, tmp_path):
    data = b"pretend model bytes"
    monkeypatch.setattr(dm, "EXPECTED_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen(data))
    dest = str(tmp_path / "m.task")
    assert dm.ensure_model(dest, quiet=True) == dest
    assert open(dest, "rb").read() == data and dm.is_valid(dest)
    assert [p.name for p in tmp_path.iterdir()] == ["m.task"]


def test_wrong_checksum_is_rejected_and_nothing_is_kept(dm, monkeypatch, tmp_path):
    monkeypatch.setattr(dm.urllib.request, "urlopen", fake_urlopen(b"tampered"))
    dest = str(tmp_path / "m.task")
    with pytest.raises(RuntimeError, match="checksum"):
        dm.ensure_model(dest, quiet=True)
    assert list(tmp_path.iterdir()) == []


def test_network_failure_gives_manual_instructions(dm, monkeypatch, tmp_path):
    def boom(url, timeout=30): raise urllib.error.URLError("offline")
    monkeypatch.setattr(dm.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="manually"):
        dm.ensure_model(str(tmp_path / "m.task"), quiet=True)
    assert list(tmp_path.iterdir()) == []


def test_existing_valid_file_is_not_redownloaded(dm, monkeypatch, tmp_path):
    data = b"x"
    monkeypatch.setattr(dm, "EXPECTED_SHA256", hashlib.sha256(data).hexdigest())
    dest = tmp_path / "m.task"; dest.write_bytes(data)
    monkeypatch.setattr(dm.urllib.request, "urlopen", lambda *a, **k: pytest.fail("should not download"))
    dm.ensure_model(str(dest), quiet=True)


def test_model_url_is_the_official_float16_v1(dm):
    assert dm.MODEL_URL == ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                            "hand_landmarker/float16/1/hand_landmarker.task")


# ── run.py / requirements ──────────────────────────────────────────────────
def test_every_requirement_is_pinned():
    lines = [l.strip() for l in open(os.path.join(ROOT, "requirements.txt")) if l.strip() and not l.startswith("#")]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][\w.]*", l) for l in lines), lines


def test_run_py_checks_the_same_packages_as_requirements():
    run = load("run.py", "run_entry")
    pinned = {re.split("==", l.strip())[0].lower()
              for l in open(os.path.join(ROOT, "requirements.txt")) if l.strip() and not l.startswith("#")}
    assert {v.lower() for v in run.REQUIRED_PACKAGES.values()} == pinned


def test_run_py_reports_missing_package_and_old_python(monkeypatch, capsys):
    run = load("run.py", "run_entry2")
    monkeypatch.setattr(run, "REQUIRED_PACKAGES", {"definitely_missing_mod_xyz": "some-pkg"})
    with pytest.raises(SystemExit):
        run.check_packages()
    assert "pip install -r requirements.txt" in capsys.readouterr().out
    monkeypatch.setattr(run, "MIN_PYTHON", (99, 0))
    with pytest.raises(SystemExit):
        run.check_python()


def test_no_personal_data_or_user_config_in_repo():
    tracked_cfg = os.path.join(ROOT, "hand_gesture_system", "config.json")
    gi = open(os.path.join(ROOT, ".gitignore")).read()
    assert "hand_gesture_system/config.json" in gi and "hand_landmarker.task" in gi
    for dirpath, _, files in os.walk(ROOT):
        if any(x in dirpath for x in (".git", "venv", "__pycache__", ".pytest_cache", ".ruff_cache")):
            continue
        for f in files:
            if f != "test_tools_and_setup.py" and f.endswith((".py", ".json", ".md", ".txt", ".yml")) and os.path.join(dirpath, f) != tracked_cfg:
                text = open(os.path.join(dirpath, f), encoding="utf-8", errors="ignore").read()
                assert ("10.53." + "220") not in text, f
