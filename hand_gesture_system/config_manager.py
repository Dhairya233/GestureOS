"""
config_manager.py
------------------
JSON-backed configuration manager for the Hand Gesture System.

Responsibilities
-----------------
- Owns the canonical default configuration dictionary.
- Exposes load_config() which reads config.json on disk, or creates it from
  defaults if the file is absent or malformed.
- Exposes save_config(config_dict) which writes the current state back to
  config.json atomically (write to tmp -> rename) so a crash never corrupts
  the config file.

Schema (all keys are always present after load_config())
---------------------------------------------------------
Feature toggles (bool):
    enable_mouse      -- Air mouse + click + drag + right-click
    enable_scroller   -- Open-palm proportional scroll
    enable_media      -- Play/pause + volume knob
    enable_launcher   -- Left-hand finger-count app launcher
    enable_canvas     -- Air Canvas drawing layer
    enable_modifier   -- Left-fist Ctrl key hold

Sensitivity settings:
    mouse_smoothing   (int/float)  -- Exponential smoothing divisor (1-15)
    scroll_speed      (float)      -- Scroll tick magnitude (0.01-1.0)
    volume_delay      (float)      -- Seconds between volume steps (0.05-0.5)
    screen_reduction  (float)      -- Active-zone reduction factor (0.2-1.0)

Other:
    swap_handedness   (bool)       -- Swap MediaPipe's Left/Right label (default true)
    kill_frames       (int)        -- Frames the thumbs-up must be held to quit

Canvas settings:
    canvas_active_color  (str)  -- Active brush colour name ("green", "red", etc.)
    canvas_brush_thickness (int) -- Brush stroke width in pixels (1-20)

App paths (dict):
    app_paths["2_fingers"] -- command / URL launched by 2-finger hold
    app_paths["3_fingers"] -- command / URL launched by 3-finger hold
    app_paths["4_fingers"] -- command / URL launched by 4-finger hold

Removed features: see RETIRED_KEYS below.
"""

import json
import os
import tempfile

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# The defaults live in config.default.json (single source of truth, shipped in
# the repo).  The user's own config.json is created from it on first run and is
# git-ignored, so personal settings never get committed.
DEFAULTS_PATH = os.path.join(os.path.dirname(__file__), "config.default.json")


def _read_defaults() -> dict:
    with open(DEFAULTS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


DEFAULTS: dict = _read_defaults()

# Keys written by older versions for features that were removed (Music
# Studio, Gesture Games, Mobile Lens, Custom Shortcuts).  Old config files
# that still contain them load fine; the keys are dropped and disappear from
# disk the next time the config is saved.
RETIRED_KEYS: frozenset = frozenset({
    "piano_keys", "drum_sticks", "active_exclusive_mode",
    "camera_source_type", "camera_ip_url", "camera_link_type",
    "mobile_ip_address", "custom_shortcuts",
    # settings that nothing in the engine ever read
    "stability_frames", "deadzone", "min_detect", "min_track", "camera_index",
})


def load_config() -> dict:
    """
    Load the configuration from config.json.

    If the file does not exist, or if it is empty / malformed, the defaults
    are written to disk and returned.  Any keys present in DEFAULTS but
    absent from the file are back-filled so callers always receive a complete
    dictionary.

    Returns
    -------
    dict
        A copy of the configuration with all expected keys guaranteed.
    """
    import copy
    config = copy.deepcopy(DEFAULTS)

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                on_disk = json.load(fh)
            for key, value in on_disk.items():
                if key in RETIRED_KEYS:
                    continue
                if (
                    key in config
                    and isinstance(config[key], dict)
                    and isinstance(value, dict)
                ):
                    config[key].update(value)
                else:
                    config[key] = value
        except (json.JSONDecodeError, OSError):
            print("[CONFIG] config.json is malformed — restoring defaults.")
            save_config(config)
    else:
        save_config(config)

    return config


def save_config(config_dict: dict) -> None:
    """
    Persist config_dict to config.json using an atomic write strategy:
    write to a temporary file in the same directory, then replace the
    target so a crash during the write never leaves a half-written file.

    Parameters
    ----------
    config_dict : dict
        The configuration state to persist.
    """
    dir_name = os.path.dirname(CONFIG_PATH)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config_dict, fh, indent=4)
        os.replace(tmp_path, CONFIG_PATH)
    except OSError as exc:
        print(f"[CONFIG] Failed to save config: {exc}")
