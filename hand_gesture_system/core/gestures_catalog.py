"""
core/gestures_catalog.py
------------------------
Single source for the in-app cheat-sheet.  Every entry was written from the
controller that implements it (see features/*); if a controller changes,
change it here too.  `pose` is a key of core.poses.POSES so the diagram matches
the synthetic hands used in tests.

Hands are the PHYSICAL hand (the app mirrors the camera and, by default,
corrects MediaPipe's label swap; see the `swap_handedness` setting).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GestureInfo:
    key: str
    title: str
    hand: str            # "Right", "Left" or "Both"
    pose: str            # key of core.poses.POSES
    does: str
    flag: str            # config key that enables it; '' = always on
    pinch_to: int | None = None   # fingertip landmark the thumb touches (diagram)


CATALOG = (
    # ── Right hand ───────────────────────────────────────────────────────
    GestureInfo("mouse", "Move cursor", "Right", "point",
                "Index finger up, middle finger down, pinky curled. The index "
                "tip steers the cursor inside the cursor area.", "enable_mouse"),
    GestureInfo("click", "Click", "Right", "peace",
                "Touch thumb to middle fingertip.", "enable_click", 12),
    GestureInfo("rclick", "Right click", "Right", "three",
                "Touch thumb to ring fingertip.", "enable_rclick", 16),
    GestureInfo("drag", "Drag", "Right", "point",
                "Pinch thumb to index and hold; move; open to drop.",
                "enable_drag", 8),
    GestureInfo("scroll", "Scroll", "Right", "four",
                "Open palm. Raise it into the top band to scroll up, lower it "
                "into the bottom band to scroll down.", "enable_scroller"),
    GestureInfo("edge", "Switch window", "Right", "four",
                "Open palm at the far left (previous) or far right (next) of "
                "the frame. Hold to keep cycling.", "enable_edge"),
    GestureInfo("draw", "Air canvas", "Right", "point",
                "Index only draws. Index and middle together hover and pick a "
                "colour in the top strip.", "enable_canvas"),
    GestureInfo("quit", "Quit", "Right", "thumbs_up",
                "Thumb straight up, fingers curled. Hold until the bar fills.",
                "enable_kill"),
    # ── Left hand ────────────────────────────────────────────────────────
    GestureInfo("playpause", "Play / pause", "Left", "peace",
                "Peace sign.", "enable_media"),
    GestureInfo("mute", "Mute", "Left", "point",
                "Index finger only.", "enable_media"),
    GestureInfo("volume", "Volume", "Left", "point",
                "Pinch thumb to index in the left zone (down) or right zone "
                "(up) of the frame.", "enable_volume", 8),
    GestureInfo("launch2", "Launch: 2 fingers", "Left", "peace",
                "Hold two fingers up (thumb tucked). Opens the target set in "
                "Settings.", "enable_launcher"),
    GestureInfo("launch3", "Launch: 3 fingers", "Left", "three",
                "Hold three fingers up.", "enable_launcher"),
    GestureInfo("launch4", "Launch: 4 fingers", "Left", "four",
                "Hold four fingers up (thumb tucked).", "enable_launcher"),
    GestureInfo("ctrl", "Hold Ctrl", "Left", "fist",
                "Make a fist. Ctrl stays down while the fist holds.",
                "enable_modifier"),
    # ── Both hands ───────────────────────────────────────────────────────
    GestureInfo("arm", "Arm / disarm", "Both", "open",
                "Show both open palms and hold. Always active.", ""),
    GestureInfo("transform", "Transform box", "Both", "point",
                "Pinch thumb and index on both hands; spread, bring together "
                "or rotate. Display only: sends nothing to the PC.",
                "", 8),
)
