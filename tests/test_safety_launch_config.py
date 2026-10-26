import json
import pytest
from core.safety import SafeRunner
from features.app_launcher.launch import LaunchError, launch, parse_target


def test_one_failing_controller_does_not_stop_others():
    logs = []
    r = SafeRunner(logs.append, repeat_every=3)
    def boom(): raise ValueError("bad")
    assert r.run("mouse", boom) is None
    assert r.run("scroll", lambda: 42) == 42
    assert len(logs) == 1 and "[mouse] ValueError: bad" in logs[0]
    for _ in range(5):
        r.run("mouse", boom)
    assert r.failures["mouse"] == 6
    assert len(logs) == 3          # 1st, then 3rd and 6th repeat (rate-limited)


def test_failure_streak_resets_after_success():
    logs = []
    r = SafeRunner(logs.append)
    ok = {"v": False}
    def f():
        if not ok["v"]: raise RuntimeError("x")
    r.run("a", f); ok["v"] = True; r.run("a", f); ok["v"] = False; r.run("a", f)
    assert len(logs) == 2


@pytest.mark.parametrize("bad", ["", "   ", "calc\nnotepad", "a" * 600,
                                 "file:///C:/Windows/win.ini", "javascript:alert(1)", "ftp://x/y"])
def test_launch_rejects_bad_targets(bad):
    with pytest.raises(LaunchError):
        parse_target(bad)


def test_urls_use_browser_and_www_gets_https():
    opened = []
    assert launch("https://example.com/a?b=1&c=2", opener=opened.append)
    launch("www.example.com", opener=opened.append)
    assert opened == ["https://example.com/a?b=1&c=2", "https://www.example.com"]


def test_command_runs_without_shell_and_metacharacters_are_inert():
    calls = []
    def runner(argv, shell): calls.append((argv, shell))
    launch("tool --flag & calc", runner=runner, which=lambda n: "/bin/" + n)
    argv, shell = calls[0]
    assert shell is False
    assert argv == ["/bin/tool", "--flag", "&", "calc"]   # '&' is just an argument


def test_missing_program_is_an_error_not_a_crash():
    with pytest.raises(LaunchError):
        launch("definitely-not-a-program-xyz", which=lambda n: None, runner=lambda *a, **k: None)


def test_launcher_does_nothing_while_disarmed(pg, monkeypatch):
    from features.app_launcher import launcher_handler as lh
    called = []
    monkeypatch.setattr(lh, "launch", lambda t: called.append(t) or t)
    a = lh.AppLauncher()
    a._launch(3, {"app_paths": {"3_fingers": "https://example.com"}})
    assert called == []
    from core import os_input
    os_input.ARMING.set_armed(True)
    a._launch(3, {"app_paths": {"3_fingers": "https://example.com"}})
    assert called == ["https://example.com"]


def test_old_config_with_retired_keys_loads_and_is_cleaned(tmp_path, monkeypatch):
    import config_manager as cm
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "scroll_speed": 0.3, "piano_keys": {"a": "x"}, "drum_sticks": {}, "active_exclusive_mode": "piano",
        "mobile_ip_address": "http://10.0.0.1:8080", "custom_shortcuts": {"fist": "enter"},
        "stability_frames": 9, "min_detect": 0.1}))
    monkeypatch.setattr(cm, "CONFIG_PATH", str(path))
    cfg = cm.load_config()
    assert cfg["scroll_speed"] == 0.3 and cfg["swap_handedness"] is True
    assert not (set(cm.RETIRED_KEYS) & set(cfg))
    cm.save_config(cfg)
    assert not (set(cm.RETIRED_KEYS) & set(json.loads(path.read_text())))


def test_default_config_file_has_no_personal_data():
    import config_manager as cm
    text = json.dumps(cm.DEFAULTS)
    assert "192.168" not in text and "10.53" not in text


def test_peace_sign_never_launches_the_two_finger_target(pg, monkeypatch):
    """Peace sign = play/pause (media). Holding it must not also launch the 2-finger app."""
    from conftest import make_hand
    from features.app_launcher import launcher_handler as lh
    from core import os_input
    called = []
    monkeypatch.setattr(lh, "launch", lambda t: called.append(t) or t)
    os_input.ARMING.set_armed(True)
    a = lh.AppLauncher()
    cfg = {"enable_media": True, "app_paths": {"2_fingers": "https://example.com"}}
    for _ in range(40):
        a.update([make_hand("Left", pose="peace")], cfg)
    assert called == []
    # with media switched off the peace sign is just "two fingers" again
    a2 = lh.AppLauncher()
    for _ in range(40):
        a2.update([make_hand("Left", pose="peace")], dict(cfg, enable_media=False))
    assert called == ["https://example.com"]


def test_three_fingers_still_launch_while_media_is_on(pg, monkeypatch):
    from conftest import make_hand
    from features.app_launcher import launcher_handler as lh
    from core import os_input
    called = []
    monkeypatch.setattr(lh, "launch", lambda t: called.append(t) or t)
    os_input.ARMING.set_armed(True)
    a = lh.AppLauncher()
    for _ in range(40):
        a.update([make_hand("Left", pose="three")], {"enable_media": True, "app_paths": {"3_fingers": "https://example.com"}})
    assert called == ["https://example.com"]
