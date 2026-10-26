"""
core/timing.py
--------------
Per-stage frame timings recorded by the vision engine (a few perf_counter calls
per frame) and read by tools/benchmark.py.  Stages: capture, mediapipe,
classify, controllers, overlay, frame (the whole _process_frame).
"""

from __future__ import annotations

from collections import deque

import numpy as np


class StageTimings:
    def __init__(self, maxlen: int = 5000) -> None:
        self._maxlen = maxlen
        self._d: dict[str, deque] = {}

    def record(self, stage: str, seconds: float) -> None:
        self._d.setdefault(stage, deque(maxlen=self._maxlen)).append(seconds)

    def clear(self) -> None:
        self._d.clear()

    def summary(self, skip: int = 0) -> dict:
        """{stage: {'mean_ms', 'p95_ms', 'n'}}, ignoring the first `skip` samples (warm-up)."""
        out = {}
        for stage, dq in list(self._d.items()):
            v = np.array(list(dq)[skip:], dtype=float) * 1000.0
            if len(v):
                out[stage] = {"mean_ms": float(v.mean()), "p95_ms": float(np.percentile(v, 95)), "n": int(len(v))}
        return out
