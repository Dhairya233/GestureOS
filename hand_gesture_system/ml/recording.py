"""
ml/recording.py
---------------
The logic of the recording tool, separated from the camera and the window so it
can be tested: which frames are saved, and why others are skipped.

A frame is saved only when exactly ONE hand is visible.  With two hands the
label would be ambiguous (which hand is making the labelled gesture?), so those
frames are counted as skipped instead of guessed at.
"""

from __future__ import annotations

from collections import Counter

from .dataset import CsvWriter, row_for_hand
from .labels import CLASSES


class Recorder:
    def __init__(self, writer: CsvWriter, session: str) -> None:
        self.writer, self.session = writer, session
        self.recording = False
        self.label = CLASSES[0]
        self.saved = Counter()
        self.skipped = Counter()

    def set_label(self, label: str) -> None:
        if label not in CLASSES:
            raise ValueError(f"unknown class {label!r}")
        self.label = label

    def toggle(self) -> bool:
        self.recording = not self.recording
        return self.recording

    def feed(self, hands, t: float, frame_size) -> str:
        """Handle one camera frame.  Returns 'paused', 'no_hand', 'multiple_hands' or 'saved'."""
        if not self.recording:
            return "paused"
        if not hands:
            self.skipped["no_hand"] += 1
            return "no_hand"
        if len(hands) > 1:
            self.skipped["multiple_hands"] += 1
            return "multiple_hands"
        self.writer.write(row_for_hand(self.label, self.session, t, hands[0], frame_size))
        self.saved[self.label] += 1
        return "saved"
