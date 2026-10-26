"""
tools/download_model.py
-----------------------
Fetch the MediaPipe HandLandmarker model and check it against a pinned
SHA-256 before using it.  Standard library only, so it runs before any
`pip install`.

    python tools/download_model.py            # download if missing/corrupt
    python tools/download_model.py --verify   # only check the existing file

Why a checksum?  The model is a binary the app loads into native code.  Pinning
its hash means a truncated download or a swapped file is rejected instead of
being silently used.

Where the hash comes from: Google does not publish a checksum for this file.
EXPECTED_SHA256 is the hash of the copy that shipped in this repo originally
(a valid 2023-04-26 .task bundle that loads in MediaPipe).  If you ever
re-download from Google and the hash differs, check the file loads, then update
the constant in one commit.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
EXPECTED_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"

DEFAULT_DEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hand_gesture_system",
    "hand_landmarker.task",
)


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: str) -> bool:
    return os.path.isfile(path) and sha256_of(path) == EXPECTED_SHA256


def ensure_model(dest: str = DEFAULT_DEST, quiet: bool = False) -> str:
    """Return `dest` once it holds a verified model, downloading if needed.

    Raises RuntimeError with a human-readable fix if that is not possible.
    """
    say = (lambda *_: None) if quiet else print

    if is_valid(dest):
        return dest
    if os.path.exists(dest):
        say(f"[MODEL] {dest} fails the checksum check; re-downloading.")

    say("[MODEL] Downloading hand_landmarker.task (~7.8 MB) ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".tmp")
    os.close(fd)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        got = sha256_of(tmp)
        if got != EXPECTED_SHA256:
            raise RuntimeError(
                "Downloaded model has an unexpected checksum.\n"
                f"  expected {EXPECTED_SHA256}\n  got      {got}\n"
                "Delete it and try again; if it keeps happening, see the note in "
                "tools/download_model.py."
            )
        os.replace(tmp, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Could not download the model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {dest}"
        ) from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    say("[MODEL] OK - checksum verified.")
    return dest


def main(argv: list[str]) -> int:
    dest = DEFAULT_DEST
    try:
        if "--verify" in argv:
            ok = is_valid(dest)
            print(f"[MODEL] {dest}: {'OK' if ok else 'MISSING or CORRUPT'}")
            return 0 if ok else 1
        ensure_model(dest)
        return 0
    except RuntimeError as exc:
        print(f"[MODEL] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
