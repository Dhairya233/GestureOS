# Learned gesture classifier (optional)

An experiment: can a small model trained on MediaPipe landmarks name hand poses as well as the hand-written geometry rules, and what does it cost? Everything here is reproducible from the code in this repository. **There are no results in this repository yet**; the numbers appear when you record your own data and run the commands below (see "Status").

## What it is, and what it is not
- `classifier: rules | ml` in Settings (default `rules`). It decides what the **Live tab and the overlay say** your pose is.
- The controllers that move the cursor, click, scroll and so on **always use the geometry rules**. Letting a statistical model trigger OS actions is a separate decision: it can be wrong in ways rules cannot, and the app can click and launch programs. This repo keeps that boundary deliberately.
- A missing, corrupt or mismatched model never stops the app: it falls back to the rules and says why (Settings page, Diagnostics log).

## Reproduce it
```
pip install -r requirements-ml.txt                       # scikit-learn + matplotlib (training only)
python tools/record_gestures.py --session desk_evening   # repeat for 5+ sittings (see the guide)
python tools/train_classifier.py data/*.csv              # evaluate, plot, write the model
python tools/benchmark.py --source camera                # per-stage latency, with and without ML
```
Outputs: `docs/media/` (confusion matrix, accuracy and robustness comparison, per-class F1, latency bars, `results.md` ready to paste into the README, `latency.md`), and `hand_gesture_system/models/gesture_model.npz` (far under 1 MB). Then set Pose recognition to `ml` in Settings.

Recorded CSVs hold your hand coordinates (no images) and live in `data/`, which is git-ignored. Treat them as personal data.

## Recording guide
The model can only be as good as the variety in what you record. Each **session** is one sitting; the evaluation holds out whole sessions, so a new session name means a new sitting.
- **Sessions:** at least 5, ideally on different days and in different rooms. With fewer than 3 the tool prints a warning that the result is unreliable; with fewer than 2 it refuses (no held-out session exists).
- **Lighting:** daylight, evening lamp, dim, backlit by a window. Do not record all in one.
- **Distance:** close (hand fills a third of the frame), arm's length, and further. Hand size in pixels changes a lot and landmark noise grows when it is small.
- **Rotation:** tilt the hand a little either way; do not hold it perfectly upright. Rotate the wrist between recordings.
- **Hands:** both, in every session. Left hands are mirrored to share one model, so both help.
- **Per gesture:** 15 to 30 seconds each (about 450 to 900 frames at 30 fps). Move slightly while recording. Frames next to each other are near-duplicates, so more seconds of the same pose adds little; more variety adds a lot.
- **`none`:** record it properly: relaxed hand, half-closed hand, transitions between gestures, a hand resting on the keyboard. It must contain the in-between poses the app sees all day.
- Only frames with exactly one hand are saved.

## Preprocessing (ml/features.py)
1. **Translate to the wrist.** Position in the frame must not matter.
2. **Divide by hand size** (wrist to middle knuckle). Distance from the camera must not matter. The same ratio the rules use for pinches.
3. **Mirror left hands** to look like right hands: one model, twice the data per class.
4. **Rotate** so wrist-to-middle-knuckle points up. Tilting must not matter. Switchable; the evaluation includes a model without it so you can see what it buys.
5. **Features (50):** the 40 normalised coordinates of landmarks 1 to 20, plus the 10 pairwise distances between fingertips. Coordinates give the shape; tip distances are rotation-invariant and make a pinch easy to read.

Depth (z) is not used: it is the noisiest MediaPipe output and nothing else in the app uses it. All of this is covered by tests (rotation, scale and mirror invariance), and one of those tests caught a sign error in my first version of step 4 (see "Found while building").

## Models (fixed, nothing tuned)
Logistic regression, a random forest (200 trees) and an MLP (one hidden layer of 32), all with seed 0 and scikit-learn defaults otherwise. The hyper-parameters were fixed before any data existed and are never adjusted on the evaluation sessions. That is deliberate: tuning on the held-out sessions would make them stop being held out.

The deployable model is exported as plain arrays (`.npz`, loaded with `allow_pickle=False`) and run with numpy, so the app needs no scikit-learn and a downloaded model file cannot run code (a pickle can). Only logistic regression and the MLP can be exported this way; the forest is evaluated but not deployed.

## Evaluation (ml/evaluate.py)
- **Leave-one-session-out.** For each session, train on all the others and test on that one. Why not a random split: consecutive frames are almost the same image, so a random split puts a near-copy of every test frame into the training set and the accuracy measures memory, not generalisation. Holding out a whole session also holds out its room, lighting and posture, which is what a new day looks like.
- Scaling, normalisation statistics and weights are fitted on the training sessions of each fold only.
- **Reported:** accuracy, macro-F1, per-class precision, recall and F1, a confusion matrix (pooled over folds), and the per-session accuracies (min to max), because with a handful of sessions the spread between sessions is the honest measure of uncertainty.
- **Baseline:** the rule-based classifier scored on exactly the same held-out frames, on the same integer pixel coordinates the app sees.
- **Robustness:** each held-out session is rotated (30, 60, 90, 180 degrees) or scaled (0.5, 0.75, 1.5) with 1 px of landmark noise. The scale test is a **simulation** of distance, not a real far-away hand. The rules are not fully rotation-invariant by design (a thumbs-up must point up), which the rotation columns will show.
- The generated `results.md` states "no clear winner" whenever the gap between two classifiers is not larger than the session-to-session spread, and lists every class where the rules beat the model.

## Performance (tools/benchmark.py)
The vision engine records its own per-stage times (capture, MediaPipe, classify, controllers, overlay, whole frame); the benchmark runs it for 300 frames after a warm-up with `classifier = rules` and `ml` and reports mean and p95 in ms plus loop FPS. With a camera, FPS is capped by the camera (usually 30), so the table also gives the headroom (1000 / mean frame time).
Not included: real operating-system call time (the benchmark runs the controllers against a fake pyautogui so it can never move your mouse), and the GUI thread.

## Honest limits
Single user, one camera, a small number of sessions, lighting-dependent. The numbers describe your data and say nothing about other people, other cameras or other rooms. Simulated distance is not real distance. Pose names are a display; nothing here is evidence that a learned classifier is safe to drive the cursor.

## Status
- Built and covered by tests: recorder core, dataset format, features, evaluation, plots, benchmark, numpy model, integration and fallback.
- Checked end to end on **synthetic** hands only (a pipeline check; `tools/train_classifier.py` watermarks its output and refuses to deploy a model from it). Synthetic scores are meaningless as results because the data is made up and easy.
- **Not yet done:** recording real sessions, the real evaluation, the real latency run on a Windows machine, and trying the recorder with a webcam. These need your hardware.

## Found while building
The rotation-normalisation step had the wrong sign on the angle, so it only worked for hands that were already upright. Unit tests on upright synthetic hands could not see it. It showed up because accuracy under rotation was not flat (70 percent at 90 degrees, then 96 percent at 180), which is impossible if the features are rotation-invariant. A test now checks invariance at six angles for every pose, and was confirmed to fail on the buggy version.
