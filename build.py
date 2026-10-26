"""
build.py
--------
Repeatable PyInstaller build script for the Hand Gesture System.

Compiles the application into a single standalone executable named
``GestureOS.exe`` with no console window.

Usage
-----
    python build.py

The finished executable will be placed in ``dist/GestureOS.exe``.

Key flags
---------
- ``--onefile``          Single-file executable (no loose DLLs).
- ``--noconsole``        Hide the terminal window at runtime.
- ``--collect-data mediapipe``
                         Bundle all MediaPipe .tflite model data that the
                         library loads at runtime.  Without this flag the
                         .exe will crash on startup on machines that do not
                         have MediaPipe installed via pip.
- ``--add-data``         Include the custom hand_landmarker.task model and
                         the default config.default.json alongside the executable.
- ``--name GestureOS``   Name of the output binary.
"""

from __future__ import annotations

import os
import sys


def build() -> None:
    """Run PyInstaller programmatically to compile GestureOS.exe."""

    try:
        import PyInstaller.__main__
    except ImportError:
        print(
            "\n[BUILD ERROR]  PyInstaller is not installed.\n"
            "               Fix:  pip install pyinstaller\n"
        )
        sys.exit(1)

    # Resolve paths relative to this script's directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(script_dir, "hand_gesture_system")
    entry_point = os.path.join(script_dir, "run.py")
    model_file = os.path.join(app_dir, "hand_landmarker.task")
    config_file = os.path.join(app_dir, "config.default.json")

    if not os.path.isfile(model_file):
        print("[BUILD ERROR]  Model missing. Run:  python tools/download_model.py")
        sys.exit(1)

    # Windows uses ';' as the path separator in --add-data; POSIX uses ':'.
    sep = ";" if sys.platform == "win32" else ":"

    args = [
        entry_point,
        "--onefile",
        "--noconsole",
        "--collect-data", "mediapipe",
        f"--add-data={model_file}{sep}.",
        f"--add-data={config_file}{sep}.",
        "--name", "GestureOS",
        f"--paths={app_dir}",
        "--noconfirm",
    ]

    print("=" * 60)
    print("  GestureOS — PyInstaller Build")
    print("=" * 60)
    print(f"  Entry point : {entry_point}")
    print(f"  Model asset : {model_file}")
    print(f"  Config file : {config_file}")
    print("  Output name : GestureOS")
    print("  Flags       : --onefile --noconsole --collect-data mediapipe")
    print("=" * 60)
    print()

    PyInstaller.__main__.run(args)

    dist_path = os.path.join(script_dir, "dist", "GestureOS.exe")
    if os.path.exists(dist_path):
        size_mb = os.path.getsize(dist_path) / (1024 * 1024)
        print()
        print("=" * 60)
        print("  BUILD SUCCESSFUL")
        print(f"  Output : {dist_path}")
        print(f"  Size   : {size_mb:.1f} MB")
        print("=" * 60)
    else:
        print()
        print("[BUILD]  PyInstaller finished but GestureOS.exe was not found.")
        print(f"[BUILD]  Expected at: {dist_path}")


if __name__ == "__main__":
    build()
