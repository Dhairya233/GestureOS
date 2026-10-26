# Manual test checklist (Windows + webcam)

Automated tests cannot see a camera or the Windows desktop. Run these by hand after each change to gesture code. Run sections A and B after changing safety or gesture code, and C to F before a release or a demo recording.

Setup: `python run.py`. Use a safe window (Notepad) for anything that types or clicks. Note anything that feels different from before.

## A. Safety (Phase 3)
- [ ] App starts **DISARMED**: sidebar says DISARMED, overlay badge is red. Make every gesture: nothing happens to the PC.
- [ ] Toggle the sidebar switch: label and overlay turn ARMED (green). Gestures now work.
- [ ] Two open palms held about 1 s toggles arm/disarm; you hear a beep; the progress bar fills. Repeat in both directions.
- [ ] **Ctrl+Alt+F12** disarms from another window (try with Notepad focused).
- [ ] Drag with a pinch, then disarm mid-drag: the mouse button is released (no stuck selection).
- [ ] Make the left fist (Ctrl held), then disarm: Ctrl is released (type a letter in Notepad: no shortcut fires).
- [ ] Quit with the window X while a fist is held: Ctrl is not stuck afterwards.
- [ ] Thumbs-up quit works while DISARMED and while ARMED.
- [ ] Move the air mouse hard into every screen corner: the app keeps running.
- [ ] **Handedness:** the overlay shows `R` at your right wrist and `L` at your left. If swapped, set `"swap_handedness": false` in `hand_gesture_system/config.json`, restart.
- [ ] Launcher: hold 2 / 3 / 4 fingers on the left hand (armed). `cmd.exe`, the web link and `code` open. While disarmed nothing opens.

## B. Geometry (Phase 4): compare with how it felt before
Rotation and distance are the point of this phase. For each gesture try: normal, hand tilted about 30 degrees each way, hand sideways, and hand close to / far from the camera.
- [ ] Air mouse: pinky curled + index up moves the cursor; tilted hand still works.
- [ ] Left click (thumb to middle), right click (thumb to ring), drag (thumb to index): same feel at near and far. **If pinches feel too easy or too hard, tell me**: thresholds are `PINCH_*` in `core/geometry.py` (derived from an assumed hand size of 130 px at 640x480, not measured on your camera).
- [ ] Drag does not flicker on and off while holding a pinch.
- [ ] Scroll: open palm (four fingers) above / below the centre band; works tilted.
- [ ] Window switch: open palm at the left / right edge (outer quarter of the frame).
- [ ] Media (left hand): peace sign = play/pause, one finger up = mute, pinch + edge zone = volume.
- [ ] App launcher finger count: 2, 3 and 4 fingers are counted correctly, **including the thumb** (the thumb rule changed from "points right" to a distance test). Check with the thumb tucked and with it out.
- [ ] Left fist holds Ctrl; releasing the fist releases Ctrl.
- [ ] Canvas: index+middle in the top colour strip picks a colour (5 equal zones, the last one clears); index alone draws.
- [ ] Two-hand pinch box (air transformer) grabs, scales and rotates.
- [ ] Thumbs-up quit: the thumb must point within about 45 degrees of straight **up** on screen (intentional). A natural slight tilt quits; a fist with the thumb pointing sideways does not.
- [ ] Try another camera resolution if you have one (phone camera app, or change the webcam mode): the scroll band, edge zones and colour strip follow the new frame size. The Diagnostics log prints "Frame size is now WxH".
- [ ] Known gap: overlay text/badges are still sized for 640x480 (the overlay redesign scales them).

## C. Camera
- [ ] Unplug the USB webcam while running: the log shows "Camera frame read failed" then "Attempting camera reconnection". Plug it back in: "Camera reconnected successfully" and tracking resumes without restarting. Nothing is stuck pressed.
- [ ] Start with no webcam attached: `python run.py` exits with a clear message (no traceback).
- [ ] Phone as a webcam: the old "Mobile Lens" tab was removed because it never connected anything. A phone works only if it appears to Windows as a normal camera (DroidCam / Iriun / Camo virtual driver). Check that it is found, and that the log prints "Frame size is now WxH" if its resolution differs from 640x480.
- [ ] Windows privacy settings blocking the camera produce the "Could not open any webcam" message.

## D. Feature pass (armed, in Notepad or a browser)
- [ ] Air mouse: move, click, right-click, drag a window, drag-select text.
- [ ] Scroll up and down on a long page, at both speed extremes of the Scroll Speed slider.
- [ ] Window switch: palm at the right edge cycles Alt+Tab, left edge Alt+Shift+Tab.
- [ ] Play/pause, mute and volume up/down with a video or music playing.
- [ ] App launcher: 2, 3 and 4 fingers start the configured targets; edit the paths in Settings (press Enter or click away), and check the new targets run. Set one path to something invalid: the log says it was not launched and nothing crashes.
- [ ] Canvas pen draws on the Live tab (dark backdrop, no camera picture); colour strip and clear work.
- [ ] Feature switches on the Live tab really disable their gestures (turn each off and try it).
- [ ] Sliders in Settings (smoothing, cursor area, scroll speed, volume repeat, quit hold) change behaviour within a few seconds.

## E. Shutdown
- [ ] Close with the window X while armed: process exits, no Python left in Task Manager, no key stuck.
- [ ] Quit in the sidebar: same.
- [ ] Thumbs-up held about 20 frames: the window closes by itself.

## F. Clean install (use a fresh folder or another PC)
- [ ] `git clone`, create a venv, `pip install -r requirements.txt` succeeds on Python 3.12 or 3.13.
- [ ] `python run.py --check` reports OK (downloads the model on first run; if the checksum check complains, see the note in `tools/download_model.py`).
- [ ] `python run.py` opens the window, DISARMED.
- [ ] Delete `hand_gesture_system/config.json`: it is recreated with defaults on the next start.

## G. UI (new look; the headless render cannot check these)
- [ ] Fonts: the window uses Segoe UI Variable (Windows 11) or falls back to Segoe UI (Windows 10). Nothing is clipped or overlapping in either.
- [ ] Display scaling at 100%, 125% and 150%: every tab still fits at the smallest window size (980 x 620); the Live skeleton view stays sharp.
- [ ] Top bar: DISARMED (grey) at start; flipping the Arm switch, showing two palms, or Ctrl+Alt+F12 changes it to ARMED (green) / DISARMED within a moment, and the switch follows.
- [ ] Live tab shows ONLY the hand skeleton on a dark backdrop (no camera picture) and keeps updating for minutes while armed.
- [ ] Diagnostics tab shows the camera picture with the overlay; leaving the tab stops it.
- [ ] While armed and controlling the cursor, Ctrl+1..5 (Live, Gestures, Settings, Whiteboard, Diagnostics) still switch tabs, and so do clicks on the sidebar.
- [ ] Live tab: skeleton follows your hands; the pose names at your wrists (R / L) match what you are doing; joints that matter for the current pose are ringed (green only when armed).
- [ ] Event list: gestures appear once per pose change (no flood); with the app disarmed, a click gesture shows "blocked (disarmed)"; when armed it shows "sent Click".
- [ ] Hold bar: appears while holding the thumbs-up (quit) or 2 / 3 / 4 fingers on the left hand, and fills in about the time the setting says.
- [ ] Unplug the camera: the top bar turns red text "Camera disconnected" and the feed shows the camera-off icon; plugging it back in restores both.
- [ ] Settings: a bad launcher target (try `file:///c:/` ) shows a red message under the field and is NOT saved; "Reset" needs two clicks.
- [ ] Gestures tab: every diagram matches the pose that really triggers the action (this is the most useful check of the whole cheat-sheet).

## H. Learned classifier (optional; needs a webcam)
- [ ] `pip install -r requirements-ml.txt`, then `python tools/record_gestures.py --session test1`: the window opens mirrored, keys 1-9 change the class, SPACE counts down and records, and q saves. Frames with two hands are skipped (counter in the final printout).
- [ ] The CSV in `data/` has one row per saved frame; pose names on the Live tab agree with what you do while recording.
- [ ] After 5+ real sessions: `python tools/train_classifier.py data/*.csv` finishes, writes `docs/media/*.png` and `results.md` (no SYNTHETIC watermark), and creates `hand_gesture_system/models/gesture_model.npz` (far under 1 MB).
- [ ] Settings > Pose recognition = `ml`: the Live tab names poses. Delete the model file: within a few seconds the Settings page shows "using rules instead" and the app keeps working.
- [ ] Controls still act on the rules: with `ml` selected, scroll / click / window switch behave exactly as before.
- [ ] `python tools/benchmark.py --source camera` runs both configurations and prints the table (the benchmark never moves your real mouse).
