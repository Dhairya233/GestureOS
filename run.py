"""
run.py - the single entry point for GestureOS (Windows).

    python run.py            start the app
    python run.py --check    verify Python, packages, model and config, then exit
                             (no camera, no window)

Steps
-----
1. Make the app package importable (hand_gesture_system/ on sys.path).
2. Check the Python version and that every required package is installed.
3. Make sure the MediaPipe model is present and matches its checksum
   (downloads it on first run).
4. Probe for a working webcam.
5. Hand over to main.main().
"""

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT, "hand_gesture_system")
MODEL_PATH = os.path.join(APP_DIR, "hand_landmarker.task")
MIN_PYTHON = (3, 10)   # the code uses `X | None` in evaluated annotations

# import name -> pip name
REQUIRED_PACKAGES = {
    "cv2": "opencv-contrib-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "PIL": "Pillow",
    "customtkinter": "customtkinter",
    "pyautogui": "pyautogui",
    "psutil": "psutil",
}


def fail(message: str) -> None:
    print(f"\n[ERROR] {message}\n")
    sys.exit(1)


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required "
            f"(you have {sys.version.split()[0]})."
        )


def check_packages() -> None:
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        fail("Missing packages: " + ", ".join(missing) +
             "\n        Fix:  pip install -r requirements.txt")


def check_model() -> None:
    if getattr(sys, "frozen", False):   # PyInstaller build bundles the model
        return
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from download_model import ensure_model
    try:
        ensure_model(MODEL_PATH)
    except RuntimeError as exc:
        fail(str(exc))


def find_webcam() -> int:
    import cv2
    for idx in (0, 1, -1):          # -1 lets Windows pick any device
        cap = cv2.VideoCapture(idx)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    print(f"[CHECK] Webcam found at index {idx}")
                    return idx
        finally:
            cap.release()
    fail("Could not open any webcam. Check the connection and Windows "
         "Settings > Privacy > Camera.")
    return -1   # unreachable


def main() -> None:
    check_only = "--check" in sys.argv[1:]

    check_python()
    check_packages()
    check_model()

    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, APP_DIR)

    if check_only:
        import config_manager
        config_manager.load_config()
        print("[CHECK] Python, packages, model and config are OK.")
        return

    webcam_index = find_webcam()

    import main as app_main
    app_main.WEBCAM_INDEX = webcam_index
    app_main.MODEL_PATH = MODEL_PATH
    app_main.main()


if __name__ == "__main__":
    main()
