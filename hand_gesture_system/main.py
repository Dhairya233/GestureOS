"""
main.py
-------
Central orchestrator for the Hand Gesture System.

Threading Architecture
──────────────────────
  Two threads run concurrently:

  ┌─────────────────────────────────────────────────────────┐
  │  VisionEngine (daemon thread)                           │
  │  ─────────────────────────────────────────────────────  │
  │  • Owns the webcam capture handle                       │
  │  • Runs all gesture feature controllers at camera FPS   │
  │  • Writes the finished RGB frame to self.latest_frame   │
  └─────────────────────────────────────────────────────────┘
                          ↕  shared numpy array (GIL-atomic write)
  ┌─────────────────────────────────────────────────────────┐
  │  Main Thread (Tkinter / CustomTkinter)                  │
  │  ─────────────────────────────────────────────────────  │
  │  • Owns the GUI event loop                              │
  │  • GestureApp.update_video_feed() polls latest_frame    │
  │    every 15 ms and pushes it to the video label         │
  │  • UI controls (switches, sliders) save config.json;    │
  │    VisionEngine re-reads it every CONFIG_RELOAD_INTERVAL│
  │    frames so changes propagate without a restart        │
  └─────────────────────────────────────────────────────────┘

Startup sequence
─────────────────
  1. Load config      — config_manager.load_config()
  2. Build controllers — HandTracker + all feature modules
  3. Create VisionEngine, start its daemon thread
  4. Create GestureApp — passes vision_engine to the window
  5. Bind WM_DELETE_WINDOW → on_closing for clean teardown
  6. app.mainloop()   — GUI runs; VisionEngine runs in parallel

Shutdown sequence (triggered by window ✕ or gesture quit)
───────────────────────────────────────────────────────────
  1. vision_engine.stop()   — sets running=False, releases cap
  2. vision_engine.join()   — waits for the thread to exit
  3. app.destroy()          — closes the Tkinter window
"""

import os
import time
import threading
import cv2
import numpy as np

import config_manager
from ui.app_window import GestureApp

from core import HandTracker, os_input, play_beep
from core.arming import ArmGesture, PanicHotkey
from core.safety import SafeRunner
from core import overlay
from core.status import STATUS, TIMELINE, GestureAnnouncer
from core.timing import StageTimings
from ml.classifier import RuleClassifier, make_classifier
from features.air_transformer  import AirTransformer
from features.tab_switcher     import EdgeJoystick
from features.system_control   import SystemController
from features.media_control    import MediaController
from features.mouse_control    import MouseController
from features.dynamic_scroller import DynamicScroller
from features.app_launcher     import AppLauncher
from features.canvas_pen       import CanvasController
from features.modifier_key     import ModifierController

# ---------------------------------------------------------------------------
# Static configuration constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

WEBCAM_INDEX: int           = 0
DETECTION_CONFIDENCE: float = 0.5
TRACKING_CONFIDENCE: float  = 0.5
MAX_HANDS: int              = 2
VERBOSE: bool               = False
MODEL_PATH: str             = os.path.join(_SCRIPT_DIR, "hand_landmarker.task")

# How often (in frames) to reload config.json from disk inside the engine.
CONFIG_RELOAD_INTERVAL: int = 30

# How many consecutive frame-read failures before the engine pauses the
# capture and waits for a reconnection attempt.  At ~30 FPS this equals
# roughly 5 seconds of continuous failures.
CAMERA_FAIL_THRESHOLD: int = 150

# Seconds to wait between reconnection attempts after a camera disconnect.
CAMERA_RECONNECT_DELAY: float = 2.0


# ===========================================================================
# Vision Engine — runs entirely on a background daemon thread
# ===========================================================================

class VisionEngine(threading.Thread):
    """
    Background thread that owns the webcam capture, runs all gesture
    controllers, annotates the frame, and exposes the finished RGB image
    via self.latest_frame for the GUI thread to consume.

    Parameters
    ----------
    tracker       : HandTracker
    controllers   : dict
        Mapping of controller names to controller instances.
    frame_w, frame_h : int
        Actual camera capture dimensions (pixels).
    """

    def __init__(
        self,
        tracker: HandTracker,
        controllers: dict,
        frame_w: int,
        frame_h: int,
    ) -> None:
        super().__init__(daemon=True, name="VisionEngine")

        self._tracker     = tracker
        self._ctrl        = controllers
        self._fw          = frame_w
        self._fh          = frame_h

        # Shared state — written by this thread, read by the GUI thread.
        # CPython's GIL makes a single object assignment atomic.
        self.latest_frame: np.ndarray | None = None       # skeleton on dark backdrop (Live)
        self.latest_camera_frame: np.ndarray | None = None  # camera + overlay (Diagnostics)
        self.want_camera_view = False                      # set by the GUI when Diagnostics is open

        # Control flags.
        # running          — set to False by stop() to exit the loop cleanly.
        # shutdown_requested — set to True when the Thumbs Up kill-switch fires;
        #                      the GUI thread's _video_loop checks this flag and
        #                      calls on_closing() on the main thread safely.
        self.running:            bool = True
        self.shutdown_requested: bool = False

        # Optional reference to the GUI app — set via set_app() after the
        # GestureApp is constructed.  Used to push diagnostic warnings.
        self._app = None

        # Safety: global armed state (disarmed at startup), arming gesture,
        # panic hotkey, and per-controller fault isolation.
        self.arming = os_input.ARMING
        self.arming.set_armed(False)
        self._was_armed: bool = False
        self._arm_gesture = ArmGesture(hold_frames=30)
        self._panic = PanicHotkey()
        self._safe = SafeRunner(self._log)

        # Display-only telemetry for the UI and overlay.
        self._announcer = GestureAnnouncer(TIMELINE)
        self.timings = StageTimings()
        self._classifier = RuleClassifier()
        self._cls_requested = "rules"
        self._fps: float = 0.0
        self._last_t = None
        self._seen_launch: float = 0.0

        # Camera failure tracking for reconnection logic.
        self._consecutive_failures: int = 0

        # Webcam capture handle — owned exclusively by this thread.
        # Wrapped in try/except so a bad index or unreachable IP stream
        # doesn't crash during construction.
        try:
            self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        except Exception as exc:
            print(f"[VisionEngine] Failed to open camera: {exc}")
            self.cap = cv2.VideoCapture()  # empty/disconnected handle

        # Mutable config snapshot reloaded periodically from disk.
        self._cfg: dict = config_manager.load_config()
        self._frame_count: int = 0

    def set_app(self, app) -> None:
        """Wire a reference to the GUI app for diagnostics logging."""
        self._app = app

    def _log(self, message: str) -> None:
        """Push a warning to the diagnostics UI (thread-safe) and stdout."""
        print(f"[VisionEngine] {message}")
        if self._app is not None:
            try:
                self._app.append_log(message)
            except Exception:
                pass

    # ── Main loop ─────────────────────────────────────────────────────────

    def _attempt_reconnect(self) -> bool:
        """
        Release the current capture and attempt to re-open the camera.

        Returns True if the reconnection succeeds (a live frame is read),
        False otherwise.
        """
        self._log("Attempting camera reconnection...")
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        try:
            self.cap = cv2.VideoCapture(WEBCAM_INDEX)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    self._log("Camera reconnected successfully.")
                    self._consecutive_failures = 0
                    return True
        except Exception as exc:
            self._log(f"Reconnection error: {exc}")

        return False

    def run(self) -> None:
        """Thread entry: run the loop, and always free held keys / the camera."""
        try:
            self._loop()
        except Exception as exc:          # last resort: never die silently
            self._log(f"FATAL: vision loop crashed: {type(exc).__name__}: {exc}")
        finally:
            os_input.release_all()
            try:
                if self.cap is not None and self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass

    def _set_frame_size(self, w: int, h: int) -> None:
        """Adopt a new frame size: zones are fractions of it, never constants."""
        self._log(f"Frame size is now {w}x{h}; zones rescaled.")
        self._fw, self._fh = w, h
        self._ctrl["scroller"].set_frame_size(h)
        self._ctrl["edge_joystick"].set_frame_size(w)
        self._ctrl["transformer"].center_x = w / 2.0
        self._ctrl["transformer"].center_y = h / 2.0
        self._ctrl["mouse"].frame_r = int(
            (1.0 - float(self._cfg.get("screen_reduction", 0.65))) * min(w, h)
        )

    def _select_classifier(self, requested: str) -> None:
        """Switch between 'rules' and 'ml' (display only).  A missing or broken model
        never stops the app: it falls back to the rules and says why, once."""
        if requested == self._cls_requested:
            return
        self._cls_requested = requested
        self._classifier, note = make_classifier(requested, log=self._log)
        STATUS.update(classifier=self._classifier.name, classifier_note=note)
        TIMELINE.add("system", f"Pose classifier: {self._classifier.name}" + (" (fallback)" if note else ""))

    def _on_arming_changed(self, armed: bool) -> None:
        """Runs on the vision thread when the armed state flips (from the GUI,
        the gesture or the hotkey).  Releases anything held and resets the
        controllers that track held state so they re-sync cleanly."""
        self._was_armed = armed
        os_input.release_all()
        self._ctrl["mouse"].reset()
        self._ctrl["modifier"].reset()
        self._log("State: ARMED - gestures now control the PC." if armed
                  else "State: DISARMED - gestures are ignored.")

    def _loop(self) -> None:
        """Camera loop: capture, then process each frame."""
        while self.running:
            # ── Safe frame acquisition ────────────────────────────────
            _t_cap = time.perf_counter()
            try:
                ret, frame = self.cap.read()
            except Exception as exc:
                self._log(f"Camera read exception: {exc}")
                ret, frame = False, None

            if not ret or frame is None:
                self._consecutive_failures += 1

                if self._consecutive_failures == 1:
                    self._log("WARNING: Camera frame read failed.")
                    TIMELINE.add("system", "Camera frame read failed")
                if self._consecutive_failures >= 3:
                    STATUS.update(camera="disconnected",
                                  camera_message="No frames from the camera; retrying")

                # After sustained failures, attempt reconnection.
                if self._consecutive_failures >= CAMERA_FAIL_THRESHOLD:
                    self._log(
                        f"Camera unavailable ({self._consecutive_failures} "
                        f"consecutive failures). Pausing for reconnect..."
                    )
                    if not self._attempt_reconnect():
                        time.sleep(CAMERA_RECONNECT_DELAY)
                    self._consecutive_failures = 0

                continue

            # Successful read — reset failure counter.
            self.timings.record("capture", time.perf_counter() - _t_cap)
            self._consecutive_failures = 0
            self._frame_count += 1

            self._safe.run("frame", self._process_frame, frame)

    def _process_frame(self, frame) -> None:
        """Detect hands, run the controllers, draw the overlay, publish."""
        # If the camera delivers a different size than we planned for (another
        # camera, a reconnect), re-derive every pixel zone from the real size.
        _t_frame = time.perf_counter()
        _fh, _fw = frame.shape[:2]
        if (_fw, _fh) != (self._fw, self._fh):
            self._set_frame_size(_fw, _fh)

        # Periodic config reload so GUI toggle changes propagate.
        if self._frame_count % CONFIG_RELOAD_INTERVAL == 0:
            self._cfg = config_manager.load_config()

            # ── Patch live-tuneable values into controllers ────────
            self._ctrl["mouse"].smoothening = float(
                self._cfg.get("mouse_smoothing", 5)
            )
            self._ctrl["mouse"].frame_r = int(
                (1.0 - float(self._cfg.get("screen_reduction", 0.65)))
                * min(self._fw, self._fh)
            )
            self._ctrl["scroller"].scroll_speed_modifier = float(
                self._cfg.get("scroll_speed", 0.1)
            )
            self._ctrl["media"].volume_repeat_sec = float(
                self._cfg.get("volume_delay", 0.15)
            )
            self._ctrl["sys_ctrl"].required_frames = int(
                self._cfg.get("kill_frames", 20)
            )
            self._tracker.swap_handedness = bool(
                self._cfg.get("swap_handedness", True)
            )
            self._select_classifier(str(self._cfg.get("classifier", "rules")))

        cfg = self._cfg

        # Mirror the frame — must happen before landmark detection.
        frame = cv2.flip(frame, 1)

        # ══════════════════════════════════════════════════════════════
        # Hand Tracking Pipeline
        # ══════════════════════════════════════════════════════════════

        # ── Hand detection ─────────────────────────────────────────────
        _t = time.perf_counter()
        all_hands = self._safe.run("tracker", self._tracker.get_landmarks, frame)
        self.timings.record("mediapipe", time.perf_counter() - _t)

        # ── Arming: panic hotkey, two-palms gesture, UI toggle ─────────
        safe = self._safe.run
        if self._panic.poll() and self.arming.set_armed(False):
            self._log("DISARMED (panic hotkey Ctrl+Alt+F12)")
        if safe("arm_gesture", self._arm_gesture.update, all_hands):
            now_armed = self.arming.toggle()
            play_beep(1500 if now_armed else 600, 120)
            self._log("ARMED (two-palms gesture)" if now_armed
                      else "DISARMED (two-palms gesture)")
        armed = self.arming.armed
        if armed != self._was_armed:
            self._on_arming_changed(armed)

        # While both palms are open (arming gesture in progress) the other
        # gestures see "no hands", which also releases any drag / held key.
        ctrl_hands = None if self._arm_gesture.engaged else all_hands
        _t_ctrl = time.perf_counter()

        # ── Feature controllers (config-gated, each isolated) ──────────
        safe("transformer", self._ctrl["transformer"].update, ctrl_hands)

        if cfg.get("enable_edge", True):
            safe("edge_joystick", self._ctrl["edge_joystick"].update, ctrl_hands)

        if cfg.get("enable_media", True) or cfg.get("enable_volume", True):
            safe("media", self._ctrl["media"].update,
                 ctrl_hands,
                 enable_media=cfg.get("enable_media", True),
                 enable_volume=cfg.get("enable_volume", True),
                 frame_width=self._fw)

        if cfg.get("enable_mouse", True):
            safe("mouse", self._ctrl["mouse"].update,
                 ctrl_hands,
                 frame_width=self._fw,
                 frame_height=self._fh,
                 enable_click=cfg.get("enable_click", True),
                 enable_rclick=cfg.get("enable_rclick", True),
                 enable_drag=cfg.get("enable_drag", True))

        if cfg.get("enable_scroller", True):
            safe("scroller", self._ctrl["scroller"].update, ctrl_hands)

        if cfg.get("enable_launcher", True):
            safe("launcher", self._ctrl["launcher"].update, ctrl_hands, cfg)

        canvas_layer = None
        if cfg.get("enable_canvas", True):
            canvas_layer = safe("canvas", self._ctrl["canvas"].update, ctrl_hands, frame)
        if canvas_layer is None:
            canvas_layer = np.zeros_like(frame)

        modifier = self._ctrl["modifier"]
        if cfg.get("enable_modifier", True):
            safe("modifier", modifier.update, ctrl_hands)
        elif modifier.is_ctrl_held:
            os_input.keyUp("ctrl")
            modifier.reset()

        hand_detected = all_hands is not None

        self.timings.record("controllers", time.perf_counter() - _t_ctrl)

        # ── Display-only state: pose names, status snapshot, timeline ──
        _t = time.perf_counter()
        gestures = self._safe.run("classifier", self._classifier.describe, all_hands) or {}
        self.timings.record("classify", time.perf_counter() - _t)
        self._announcer.update(gestures)
        self._publish_status(all_hands, gestures, armed)

        # ── Overlay (ink first so the skeleton and HUD sit on top) ─────
        # The Live tab shows only the hand skeleton on a dark backdrop
        # (design C); the camera picture is rendered only while the
        # Diagnostics tab asks for it.
        _t = time.perf_counter()
        hold_label, hold_progress = self._hold_info(all_hands)
        ctx = overlay.OverlayContext(
            armed=armed,
            hands=all_hands,
            gestures=gestures,
            arm_engaged=self._arm_gesture.engaged,
            arm_progress=self._arm_gesture.progress,
            hold_label=hold_label,
            hold_progress=hold_progress,
            fps=self._fps,
            ctrl_held=modifier.is_ctrl_held,
            mouse_rect=self._ctrl["mouse"].frame_r if cfg.get("enable_mouse", True) else None,
            scroll_lines=((self._ctrl["scroller"]._upper_boundary,
                           self._ctrl["scroller"]._lower_boundary)
                          if cfg.get("enable_scroller", True) else None),
        )

        def draw_scene(base):
            if cfg.get("enable_canvas", True):
                base = cv2.addWeighted(base, 1, canvas_layer, 1, 0)
            overlay.draw(base, ctx)
            self._ctrl["edge_joystick"].draw_osd(base)
            self._ctrl["transformer"].draw(base)
            if cfg.get("enable_canvas", True):
                canvas_ctrl = self._ctrl["canvas"]
                colours = canvas_ctrl.colors
                active = (colours.index(canvas_ctrl.draw_color)
                          if canvas_ctrl.draw_color in colours else -1)
                overlay.draw_palette(base, canvas_ctrl.palette_layout(self._fw, self._fh),
                                     canvas_ctrl.header_height(self._fh), active)
            return base

        backdrop = np.empty_like(frame)
        backdrop[:] = (19, 16, 14)   # tokens.BG #0E1013 as BGR
        scene = draw_scene(backdrop)
        camera_scene = draw_scene(frame.copy()) if self.want_camera_view else None

        self.timings.record("overlay", time.perf_counter() - _t)

        if VERBOSE and hand_detected:
            for hand_data in all_hands:
                print(f"[{hand_data['label'].upper()} HAND] {len(hand_data['landmarks'])} landmarks")

        # ── Gesture-based quit signal ───────────────────────────────
        if cfg.get("enable_kill", True) and safe("kill", self._ctrl["sys_ctrl"].update, all_hands):
            print("\n[VisionEngine] Kill-switch confirmed — shutdown_requested = True")
            self.shutdown_requested = True
            self.running = False
            return

        self.timings.record("frame", time.perf_counter() - _t_frame)

        # ── Expose finished RGB frame to the GUI thread ───────────────
        self.latest_frame = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        self.latest_camera_frame = (cv2.cvtColor(camera_scene, cv2.COLOR_BGR2RGB)
                                    if camera_scene is not None else None)

    def _hold_info(self, all_hands):
        """(label, progress 0..1) of a held pose that is about to do something."""
        sc = self._ctrl["sys_ctrl"]
        if self._cfg.get("enable_kill", True) and sc.confirmation_frames > 0:
            return "Quit: keep holding thumbs up", sc.confirmation_frames / max(sc.required_frames, 1)
        la = self._ctrl["launcher"]
        if (self._cfg.get("enable_launcher", True) and la.hold_frames > 0
                and la.current_count in (2, 3, 4)
                and all_hands and any(h["label"] == "Left" for h in all_hands)):
            return (f"Launch app {la.current_count}: hold",
                    la.hold_frames / max(la.target_hold_threshold, 1))
        return "", 0.0

    def _publish_status(self, all_hands, gestures: dict, armed: bool) -> None:
        hand_detected = all_hands is not None
        now = time.monotonic()
        if self._last_t is not None and now > self._last_t:
            inst = 1.0 / (now - self._last_t)
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_t = now
        label, progress = self._hold_info(all_hands)
        STATUS.update(armed=armed, camera="connected" if hand_detected else "no_hand",
                      camera_message="", fps=self._fps, frame_size=(self._fw, self._fh),
                      hands=len(gestures), gestures=gestures,
                      hold_label=label, hold_progress=progress,
                      classifier=self._classifier.name)
        launched = self._ctrl["launcher"].last_launch_time
        if launched and launched != self._seen_launch:
            self._seen_launch = launched
            TIMELINE.add("action", f"Launched {self._ctrl['launcher'].last_app_name}")

    # ── Shutdown ──────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Ask the loop to exit.  The capture handle is released by run() on
        the vision thread itself (releasing it here, from the GUI thread, could
        race a cap.read() in progress)."""
        self.running = False


# ===========================================================================
# Entry point
# ===========================================================================

def build_controllers(cfg: dict, frame_w: int, frame_h: int) -> dict:
    """All feature controllers, built the way the app and tools/benchmark.py need them."""
    return {
        "transformer":   AirTransformer(frame_w=frame_w, frame_h=frame_h),
        "edge_joystick": EdgeJoystick(frame_w=frame_w),
        "sys_ctrl":      SystemController(),
        "media":         MediaController(),
        "mouse":         MouseController(smoothening=float(cfg.get("mouse_smoothing", 5))),
        "scroller":      DynamicScroller(frame_h=frame_h),
        "launcher":      AppLauncher(),
        "canvas":        CanvasController(),
        "modifier":      ModifierController(),
    }


def main() -> None:
    """
    Build all subsystems, start the VisionEngine thread, launch the GUI,
    and wire the shutdown path so everything tears down cleanly.
    """

    # ------------------------------------------------------------------
    # Load initial configuration.
    # ------------------------------------------------------------------
    cfg: dict = config_manager.load_config()

    # ------------------------------------------------------------------
    # Probe the webcam once to discover the actual capture resolution.
    # ------------------------------------------------------------------
    _cap_probe = cv2.VideoCapture(WEBCAM_INDEX)
    _, _probe_frame = _cap_probe.read()
    _cap_probe.release()
    _fh, _fw = (_probe_frame.shape[:2] if _probe_frame is not None else (480, 640))

    # ------------------------------------------------------------------
    # Initialise the HandTracker (MediaPipe backend).
    # ------------------------------------------------------------------
    tracker = HandTracker(
        model_path=MODEL_PATH,
        detection_confidence=DETECTION_CONFIDENCE,
        tracking_confidence=TRACKING_CONFIDENCE,
        max_hands=MAX_HANDS,
        swap_handedness=bool(cfg.get("swap_handedness", True)),
    )

    # ------------------------------------------------------------------
    # Initialise all feature controllers.
    # ------------------------------------------------------------------
    controllers = build_controllers(cfg, _fw, _fh)

    # ------------------------------------------------------------------
    # Create and start the background vision engine thread.
    # ------------------------------------------------------------------
    vision_engine = VisionEngine(
        tracker=tracker,
        controllers=controllers,
        frame_w=_fw,
        frame_h=_fh,
    )
    vision_engine.start()
    print(f"[INFO] VisionEngine started (webcam index={WEBCAM_INDEX}).  GUI launching…")

    # ------------------------------------------------------------------
    # Create the CustomTkinter GUI window, injecting the engine.
    # ------------------------------------------------------------------
    app = GestureApp(vision_engine=vision_engine)
    vision_engine.set_app(app)

    # ------------------------------------------------------------------
    # Shutdown handler — called by the GUI window-close button or from
    # inside the vision engine's gesture-quit path.
    # ------------------------------------------------------------------
    def on_closing() -> None:
        """Stop the vision engine, wait for it, then destroy the window."""
        print("[INFO] Shutdown initiated…")
        vision_engine.stop()
        vision_engine.join(timeout=3.0)
        tracker.release()
        print("[INFO] Resources released. Goodbye!")
        try:
            app.destroy()
        except Exception:
            pass

    app.protocol("WM_DELETE_WINDOW", on_closing)

    # ------------------------------------------------------------------
    # Poll for gesture-initiated shutdown from inside the Tkinter loop.
    # ------------------------------------------------------------------
    # The VisionEngine sets shutdown_requested = True on its own thread
    # when the Thumbs Up kill-switch fires.  This poller runs on the main
    # (GUI) thread every 200 ms and calls on_closing() when it sees the
    # flag, which cleanly tears everything down through Tkinter's own
    # event loop — no cross-thread widget calls needed.
    # ------------------------------------------------------------------
    def _poll_engine_quit() -> None:
        if vision_engine.shutdown_requested:
            on_closing()
            return
        # Also catch the engine dying unexpectedly without setting the flag.
        if not vision_engine.running and not vision_engine.is_alive():
            on_closing()
            return
        app.after(200, _poll_engine_quit)

    app.after(200, _poll_engine_quit)

    # ------------------------------------------------------------------
    # Hand control to the Tkinter event loop.
    # ------------------------------------------------------------------
    app.mainloop()


if __name__ == "__main__":
    main()
