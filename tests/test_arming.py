from core.arming import Arming, ArmGesture, PanicHotkey
from conftest import make_hand


def test_starts_disarmed_and_toggles():
    a = Arming()
    assert a.armed is False
    assert a.toggle() is True and a.armed
    assert a.set_armed(True) is False       # no change
    assert a.set_armed(False) is True


def test_two_open_palms_fire_once_then_need_release():
    g = ArmGesture(hold_frames=5)
    hands = [make_hand("Left", origin=(150, 300)), make_hand("Right", origin=(450, 300))]
    fired = [g.update(hands) for _ in range(12)]
    assert fired.count(True) == 1 and fired.index(True) == 4
    assert g.engaged
    g.update(None)                           # release
    assert not g.engaged
    assert [g.update(hands) for _ in range(5)].count(True) == 1   # can fire again


def test_one_hand_or_fist_does_not_arm():
    g = ArmGesture(hold_frames=3)
    one = [make_hand("Right")]
    assert not any(g.update(one) for _ in range(10)) and not g.engaged
    fist = [make_hand("Left", open_palm=False), make_hand("Right", open_palm=False)]
    assert not any(g.update(fist) for _ in range(10)) and not g.engaged


def test_arm_gesture_is_rotation_and_scale_invariant():
    g = ArmGesture(hold_frames=2)
    for rot in (0.0, 0.8, 1.6, 3.1, -1.2):
        for sc in (0.4, 1.0, 2.0):
            hands = [make_hand("Left", rotation=rot, scale=sc, origin=(200, 300)),
                     make_hand("Right", rotation=rot, scale=sc, origin=(500, 300))]
            g.update(None)
            assert g.update(hands) is False and g.update(hands) is True, (rot, sc)


def test_panic_hotkey_edge_triggered():
    state = {0x11: False, 0x12: False, 0x7B: False}
    p = PanicHotkey(key_down=lambda vk: state[vk])
    assert p.poll() is False
    state.update({0x11: True, 0x12: True, 0x7B: True})
    assert p.poll() is True
    assert p.poll() is False                 # held: no repeat
    state.update({0x7B: False})
    p.poll()
    state.update({0x7B: True})
    assert p.poll() is True
