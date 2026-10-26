# What I found and fixed (redesign report)

Reusable in the README. Everything marked *unverified* needs a Windows run with a webcam.

## Found and fixed

| Problem found | What was done |
|---|---|
| Two advertised features (Music Studio, Gesture Games) were UI shells with no engine behind them; a phone-camera tab and a custom-shortcuts tab saved settings that nothing read; the diagnostics log printed fake "demo" cursor and FPS lines | All removed, old config files still load |
| Gestures could fire clicks, keys and app launches in any focused window with no off switch | Starts **disarmed**; arm by switch or two open palms; Ctrl+Alt+F12 disarms; held keys are always released |
| pyautogui's 0.1 s pause throttled the vision loop and its corner failsafe could kill the vision thread when the cursor reached a corner | Single input gateway (`core/os_input.py`): pause off, failsafe off, cursor clamped, replaced by the arming model |
| One exception in any controller stopped the whole engine; a crash or quit could leave Ctrl or the mouse button stuck | Each controller isolated and logged; everything held is released on disarm, quit and crash |
| App launcher ran `subprocess.Popen(cmd, shell=True)` on a string from `config.json` | No shell, validated targets, http(s) only, armed only; trust model documented |
| Finger tests assumed an upright hand (`tip.y < pip.y`, thumb by x-position) | Distance-based, rotation-invariant geometry (`core/geometry.py`) |
| Pinch thresholds were fixed pixels, so behaviour changed with distance from the camera | Ratios of hand size with hysteresis |
| Zones hard-coded for 640x480 (canvas colour strip, scroll band, edge zones) | Fractions of the real frame; engine rescales if the camera size changes |
| GUI-thread safety: the vision thread called Tk `after()` and the GUI released the camera mid-read | Log queue drained by the GUI; camera released only by its owner thread |
| Dead settings (stability frames, min detect / track, deadzone, camera index) | Removed |
| Thumbs-up quit accepted a sideways thumb | Thumb must point within 45 degrees of vertical |
| No setup story: no requirements, broken README, 7.8 MB binary and a personal LAN IP committed, no LICENSE, no tests | Pinned requirements, `python run.py` (+ `--check`), checksum-verified model download, `config.default.json`, MIT, 500+ tests, CI |
| 2,863-line `ui/app_window.py` | Split into modules of at most about 300 lines; rendered output unchanged |

## UI redesign ("Flight Recorder")

| Problem found | What was done |
|---|---|
| Generic neon look: cyan/violet accents, glow rings, emoji icons, an invented "AG SYSTEM" logo, a fake "Background Tracking Active" banner that was always green | Monochrome interface; colour only means a state (armed green, error red). No emoji; Lucide icons; no fake status |
| The app never said whether gestures were live except a small sidebar label | Persistent top bar: ARMED / DISARMED, camera state and fps, and the arm switch |
| The camera feed was hidden on the third tab, and the dashboard hints for some gestures were wrong (volume, click and right-click described incorrectly) | Live tab with a skeleton-only view on a dark backdrop (camera picture on Diagnostics), per-hand pose names, hold progress, an event list and the feature switches; cheat-sheet generated from a catalogue written from the controllers |
| No way to see what the system thought you were doing | `core/recognizer.py` names each hand's pose; `core/status.py` publishes it and an event timeline (gestures, what was sent, what was blocked while disarmed) |
| Fixed-colour OpenCV text overlay (yellow joints, badges in several colours) | `core/overlay.py`: white skeleton, rings on the joints that matter, zone guides, hold bar; scales with frame size |
| Settings were a mixed set of sliders without units; no handedness or quit-hold setting | Grouped settings with units and ranges, handedness switch, quit-hold length, inline launcher validation, two-click reset |
| "Air Canvas Studio" was a mouse-drawn whiteboard that had nothing to do with the gesture air canvas | Kept, renamed "Whiteboard", with a note saying so |
| Camera image was a PIL label (blurry when scaled) | `CTkImage`, scaled to the available space |

Display-only: the recogniser, status and overlay never trigger anything; control behaviour is unchanged. Two user-visible side effects: the old "SYSTEM READY - CENTER ZONE" banner and the always-on transformer label wording are gone.

## Learned classifier (optional)

Added: recording tool, dataset format, features, leave-one-session-out evaluation of logistic regression, random forest and MLP against the rules, robustness tests, plots, a numpy-only deployable model (no pickle), a per-stage latency benchmark built into the engine, a `classifier: rules | ml` setting with graceful fallback, and docs (docs/ML.md). Display-only: controllers still use the rules.

**What is not there: real results.** Everything was checked on synthetic hands, which says the code works and nothing about accuracy. One real bug was caught that way: the rotation normalisation had a sign error that upright tests could not see.

## Facts worth stating in the README
- Windows only. Python 3.12 or newer (pinned numpy requirement); the code itself needs 3.10.
- Pinned versions were installed and tested together on Linux with Python 3.12 and 3.13; Windows wheels exist for all of them.
- Tests cover safety logic, geometry (including rotation and scale invariance and equivalence with the old logic for upright hands), controllers with fake input, config, model download, and a GUI smoke test. They do not cover real camera input.

## Not verified (needs Windows + webcam)
1. Real MediaPipe landmarks versus my synthetic hands: cutoffs between curled and extended fingers, the thumb rule used for finger counting, and the 45-degree thumbs-up limit.
2. Pinch thresholds: derived from the old pixel values assuming a 130 px hand at 640x480, not measured.
3. Left/Right handedness label (overlay prints L / R at each wrist; `swap_handedness` in the config flips it).
4. Ctrl+Alt+F12 panic hotkey (Windows key polling), the two-palms hold time, and the beep.
5. Launching `code`, `cmd.exe` and a web link on Windows without a shell.
6. Camera unplug / replug, and other camera resolutions.
7. The model checksum is the hash of the file originally in the repo; Google publishes none and the download could not be repeated from the build environment.
8. GitHub Actions has never run (including the Windows job); `build.py` (PyInstaller) is untested and probably loses settings in one-file mode.
9. Display scaling at 125% / 150% on a real Windows display. The feed now uses `CTkImage` (sharp at any scale) and a test checks every tab at 125% in the headless renderer, but only a Windows screen can confirm it.
10. The Segoe UI Variable font (Windows 11); Windows 10 falls back to Segoe UI. The headless renders use a substitute font.
11. Gesture diagrams on the Gestures tab are drawn from an idealised hand model, written from reading the controllers; compare each with how the gesture actually works for you.
12. The recorder with a real webcam (only its core logic is tested), the real evaluation on recorded sessions, and the real latency numbers on Windows. Synthetic runs exist only as a pipeline check.

## Update after first Windows run

Reported: Gestures/Settings tabs stopped opening once gestures were controlling the PC; the Live feed showed one frame; it showed the whole camera. Changes: Live now shows only the hand skeleton on the dark backdrop and the camera picture moved to Diagnostics (rendered only while that tab is open); the feed uses a fresh image per frame instead of re-configuring one shared image; the refresh loop catches errors and always reschedules; view errors cannot block navigation; Ctrl+1..5 switch tabs by keyboard. I could not reproduce the freeze headlessly, so the root cause is not confirmed: please re-test on Windows (checklist section on Live/Diagnostics).

## Peace sign vs two-finger launcher

Holding the Left-hand peace sign (play/pause) for about 15 frames also counted as "two fingers" and launched the 2-finger target (Command Prompt by default). The launcher now ignores the media controller's peace pose while media is enabled. Side effect: index + middle up with the other fingers curled no longer launches the 2-finger app; thumb + index (an L shape) still counts as two fingers. Covered by two tests.
