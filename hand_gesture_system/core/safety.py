"""
core/safety.py
--------------
SafeRunner: call a controller so that one failure cannot take the engine down.

Exceptions are caught and logged with the controller's name.  To avoid
flooding the log at 30 fps, the first failure is logged immediately and then
only every ``repeat_every`` failures for that controller.  A successful call
resets the streak so a later, different failure is reported again.
"""

from __future__ import annotations

import traceback
from typing import Callable


class SafeRunner:
    def __init__(self, log: Callable[[str], None], repeat_every: int = 150) -> None:
        self._log = log
        self._repeat_every = repeat_every
        self.failures: dict[str, int] = {}     # total per controller
        self._streak: dict[str, int] = {}

    def run(self, name: str, fn: Callable, *args, **kwargs):
        """Return fn's result, or None if it raised."""
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.failures[name] = self.failures.get(name, 0) + 1
            n = self._streak[name] = self._streak.get(name, 0) + 1
            if n == 1 or n % self._repeat_every == 0:
                tb = traceback.extract_tb(exc.__traceback__)[-1]
                self._log(
                    f"[{name}] {type(exc).__name__}: {exc} "
                    f"({tb.filename.split('/')[-1].split(chr(92))[-1]}:{tb.lineno})"
                    + (f"  [x{n}]" if n > 1 else "")
                )
            return None
        self._streak[name] = 0
        return result
