# core/__init__.py
# Public API of the core package.  Heavy imports (cv2 / mediapipe) are lazy so
# that pure-logic modules (geometry, arming, os_input, safety) can be imported
# and tested without a camera stack installed.

__all__ = ["HandTracker", "play_beep"]


def __getattr__(name: str):
    if name == "HandTracker":
        from .hand_tracker import HandTracker
        return HandTracker
    if name == "play_beep":
        from .audio_utils import play_beep
        return play_beep
    raise AttributeError(f"module 'core' has no attribute {name!r}")
