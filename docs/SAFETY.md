# Safety model

GestureOS sends real mouse and keyboard events (clicks, Enter, Alt+Tab, Ctrl, scroll, media keys) to whichever window has focus, and can start programs. So:

## Armed / disarmed
- **Starts disarmed.** While disarmed, no OS action is sent. The camera, overlay and the quit gesture still work.
- **Arm / disarm:** the sidebar switch, or show **both open palms** for about one second (30 frames). While both palms are open the other gestures are suppressed.
- **Panic:** **Ctrl + Alt + F12**, from any window, disarms immediately (Windows).
- The video overlay always shows `ARMED` (green) or `DISARMED` (red).
- Disarming, quitting or a crash releases any held Ctrl key or mouse button.

## Why pyautogui's corner failsafe is off
pyautogui aborts when the cursor reaches a screen corner. An air mouse reaches corners all the time, and the abort would have killed the vision thread. The failsafe is replaced by the arming model, the panic hotkey and the panic gesture. Cursor targets are clamped to the screen, and pyautogui's default 0.1 s pause is disabled so it cannot throttle the loop.

## Launcher trust model
`app_paths` in `hand_gesture_system/config.json` names programs to start on a 2/3/4-finger hold. That file is **trusted and user-owned**: whatever it says runs when you make the gesture, so keep it writable only by you. Launching never uses a shell; only http(s) links and plain commands are accepted (see `features/app_launcher/launch.py`).

## Fault isolation
Each controller runs inside `SafeRunner`. An exception in one is logged (Diagnostics tab) and the others keep running.

## Handedness
MediaPipe's Left/Right label can appear swapped depending on mirroring. The overlay prints `L` / `R` at each wrist. If they are backwards for your hands, set `"swap_handedness": false` in `config.json`.
