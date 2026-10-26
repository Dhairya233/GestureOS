"""
features/app_launcher/launch.py
-------------------------------
Safe launching of the targets configured under ``app_paths`` in config.json.

Trust model
-----------
``config.json`` is a *trusted, user-owned* file: whatever it names is run when
you make the matching gesture. Keep it writable only by you, and treat a
config from someone else like a script from someone else.  Because of that,
this module still refuses to do more than it must:

* **No shell.** The old code ran ``subprocess.Popen(target, shell=True)``, so a
  value like ``calc & del ...`` was interpreted by cmd.exe.  Now a command is
  split into an argument list and started directly, so ``&``, ``|``, ``>`` and
  friends are just characters.
* **Only web links we expect.** Only http(s) URLs (or ``www.`` which becomes
  https) are opened in the browser; ``file:``, ``javascript:`` and the like are
  rejected.
* **Validation.** Empty values, control characters and absurd lengths are
  rejected before anything runs.
* **Armed only.** Nothing launches while GestureOS is disarmed.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import webbrowser
from dataclasses import dataclass

MAX_TARGET_LEN = 500


class LaunchError(ValueError):
    """The configured target is unusable (message is safe to show the user)."""


@dataclass(frozen=True)
class LaunchSpec:
    kind: str                 # "url" or "command"
    url: str = ""
    argv: tuple = ()


def parse_target(target: str) -> LaunchSpec:
    """Validate a config value and turn it into a LaunchSpec. Pure function."""
    if not isinstance(target, str) or not target.strip():
        raise LaunchError("empty launch target")
    target = target.strip()
    if len(target) > MAX_TARGET_LEN:
        raise LaunchError("launch target is too long")
    if any(ord(c) < 32 for c in target):
        raise LaunchError("launch target contains control characters")

    lower = target.lower()
    if lower.startswith(("http://", "https://")):
        return LaunchSpec("url", url=target)
    if lower.startswith("www."):
        return LaunchSpec("url", url="https://" + target)
    if "://" in target.split()[0] or lower.startswith(("javascript:", "data:", "file:")):
        raise LaunchError("only http/https links are allowed")

    try:
        argv = shlex.split(target, posix=(sys.platform != "win32"))
    except ValueError as exc:
        raise LaunchError(f"cannot parse command: {exc}") from exc
    if not argv:
        raise LaunchError("empty launch target")
    argv[0] = argv[0].strip('"')
    return LaunchSpec("command", argv=tuple(argv))


def launch(target: str, *, opener=None, runner=None, which=None) -> str:
    """Launch ``target``; returns a short label for the on-screen log.

    ``opener`` / ``runner`` / ``which`` are injectable for tests.
    """
    spec = parse_target(target)
    if spec.kind == "url":
        (opener or webbrowser.open)(spec.url)
        return spec.url

    exe = (which or shutil.which)(spec.argv[0])
    argv = list(spec.argv)
    if exe:
        argv[0] = exe
    elif os.path.exists(argv[0]) and hasattr(os, "startfile") and len(argv) == 1:
        os.startfile(argv[0])            # documents / folders (Windows)
        return spec.argv[0]
    else:
        raise LaunchError(f"program not found: {spec.argv[0]}")
    (runner or subprocess.Popen)(argv, shell=False)
    return target
