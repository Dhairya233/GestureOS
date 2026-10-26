"""
ml/dataset.py
-------------
CSV format for recorded landmarks, and the loader.

One row per frame per hand:

    label, session, t, handedness, frame_w, frame_h, x0, y0, x1, y1, ... x20, y20

* label        one of ml.labels.CLASSES
* session      free text naming one recording sitting (room, lighting, day).
               The evaluation splits by this column; see docs/ML.md for why.
* t            seconds since the session started
* handedness   'Left' / 'Right' exactly as the app reports it
* x*, y*       integer pixel coordinates in the MIRRORED frame, the same numbers
               the controllers see (core/hand_tracker.py output)

Depth (z) is not stored: nothing here uses it.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np

from .labels import CLASSES

META = ["label", "session", "t", "handedness", "frame_w", "frame_h"]
COORDS = [f"{a}{i}" for i in range(21) for a in "xy"]
COLUMNS = META + COORDS


class DatasetError(ValueError):
    pass


def row_for_hand(label: str, session: str, t: float, hand: dict, frame_size) -> list:
    """Build one CSV row from an app hand dict {'label', 'landmarks'}."""
    if label not in CLASSES:
        raise DatasetError(f"unknown class {label!r}")
    xy = [v for lm in hand["landmarks"] for v in (int(lm[1]), int(lm[2]))]
    return [label, session, f"{t:.3f}", hand["label"], frame_size[0], frame_size[1], *xy]


class CsvWriter:
    """Append rows to a CSV (creates the header if the file is new)."""

    def __init__(self, path: str) -> None:
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new:
            self._w.writerow(COLUMNS)
        self.count = 0

    def write(self, row: list) -> None:
        self._w.writerow(row)
        self.count += 1

    def close(self) -> None:
        self._f.close()


@dataclass
class Dataset:
    labels: np.ndarray      # (n,) str
    sessions: np.ndarray    # (n,) str
    hands: np.ndarray       # (n,) 'Left'/'Right'
    points: np.ndarray      # (n, 21, 2) float pixel coordinates
    t: np.ndarray           # (n,)

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, mask) -> "Dataset":
        return Dataset(self.labels[mask], self.sessions[mask], self.hands[mask],
                       self.points[mask], self.t[mask])

    def lms(self, i: int) -> list:
        """Row i as the app's landmark tuples [(id, x, y, 0.0), ...]."""
        return [(k, int(self.points[i, k, 0]), int(self.points[i, k, 1]), 0.0) for k in range(21)]


def load(paths) -> Dataset:
    """Read one or more CSVs written by the recorder (or the synthetic generator)."""
    labels, sessions, hands, pts, ts = [], [], [], [], []
    for path in ([paths] if isinstance(paths, str) else paths):
        with open(path, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            if header != COLUMNS:
                raise DatasetError(f"{path}: unexpected header (expected {len(COLUMNS)} columns starting "
                                   f"with {META})")
            for n, row in enumerate(rd, start=2):
                try:
                    if len(row) != len(COLUMNS):
                        raise ValueError("wrong number of columns")
                    if row[0] not in CLASSES:
                        raise ValueError(f"unknown class {row[0]!r}")
                    if row[3] not in ("Left", "Right"):
                        raise ValueError(f"bad handedness {row[3]!r}")
                    xy = np.array(row[6:], dtype=np.float64)
                    if not np.isfinite(xy).all():
                        raise ValueError("non-finite coordinate")
                    ts.append(float(row[2]))
                except ValueError as exc:
                    raise DatasetError(f"{path}, line {n}: {exc}") from exc
                labels.append(row[0]); sessions.append(row[1]); hands.append(row[3])
                pts.append(xy.reshape(21, 2))
    if not labels:
        raise DatasetError("no rows found")
    return Dataset(np.array(labels), np.array(sessions), np.array(hands),
                   np.array(pts), np.array(ts))
