from core import os_input


def names(pg):
    return [c[0] for c in pg.calls]


def test_disarmed_drops_all_actions(pg):
    os_input.moveTo(10, 10); os_input.click(); os_input.rightClick()
    os_input.scroll(3); os_input.press("a"); os_input.hotkey("alt", "tab")
    os_input.mouseDown(); os_input.keyDown("ctrl")
    assert names(pg) == []


def test_armed_sends_actions(pg):
    os_input.ARMING.set_armed(True)
    os_input.click(); os_input.press("playpause"); os_input.hotkey("alt", "tab")
    assert names(pg) == ["click", "press", "hotkey"]


def test_move_is_clamped_to_screen(pg):
    os_input.ARMING.set_armed(True)
    os_input.moveTo(99999, -50)
    assert pg.calls[-1] == ("moveTo", (1919, 0))   # never the corner-failsafe pixel issue


def test_release_works_while_disarmed_and_only_for_held(pg):
    os_input.ARMING.set_armed(True)
    os_input.keyDown("ctrl"); os_input.mouseDown()
    os_input.ARMING.set_armed(False)
    os_input.release_all()
    assert ("keyUp", ("ctrl",)) in pg.calls and ("mouseUp", ()) in pg.calls
    pg.calls.clear()
    os_input.release_all()            # nothing held now -> no spurious events
    assert pg.calls == []


def test_keyup_without_keydown_is_noop(pg):
    os_input.keyUp("ctrl"); os_input.mouseUp()
    assert pg.calls == []


def test_pyautogui_defaults_are_neutralised():
    """Regression for the corner-failsafe crash and the 0.1 s throttle."""
    import types
    fake = types.SimpleNamespace(PAUSE=0.1, FAILSAFE=True, size=lambda: (10, 10))
    import sys
    sys.modules["pyautogui"] = fake
    try:
        os_input.set_backend(None)
        os_input._pg()
        assert fake.PAUSE == 0 and fake.FAILSAFE is False
    finally:
        del sys.modules["pyautogui"]
        os_input.set_backend(None)
