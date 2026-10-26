"""
ml/labels.py
------------
The gesture classes, and the mapping between them and the strings the
rule-based recogniser (core/recognizer.py) prints.

'none' means "no gesture I know": a relaxed hand, a half-formed pose, anything
transitional.  It is a real class on purpose: a classifier forced to pick one of
the named gestures for every frame would be wrong on all the in-between frames.
"""

from __future__ import annotations

CLASSES = ("open_palm", "fist", "point", "two_fingers", "three_fingers",
           "four_fingers", "thumbs_up", "pinch", "none")

_RULE_TO_CLASS = {
    "Open palm": "open_palm", "Fist": "fist", "Point": "point",
    "Two fingers": "two_fingers", "Three fingers": "three_fingers",
    "Four fingers": "four_fingers", "Thumbs up": "thumbs_up", "Unknown": "none",
}
# The recogniser's own wording, used when the ML model names a class.
CLASS_TO_DISPLAY = {
    "open_palm": "Open palm", "fist": "Fist", "point": "Point",
    "two_fingers": "Two fingers", "three_fingers": "Three fingers",
    "four_fingers": "Four fingers", "thumbs_up": "Thumbs up", "pinch": "Pinch",
    "none": "Unknown",
}


def rule_label_to_class(text: str) -> str:
    """'Pinch (index)' -> 'pinch', 'Open palm' -> 'open_palm', unknown -> 'none'."""
    if text.startswith("Pinch"):
        return "pinch"
    return _RULE_TO_CLASS.get(text, "none")
