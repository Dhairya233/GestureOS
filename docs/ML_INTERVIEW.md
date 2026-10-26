# Interview questions about the learned classifier

Honest framing for all of them: the pipeline is built and tested; the real numbers come from my own recordings, so quote those, not these answers' placeholders.

**1. Why did you split by session instead of randomly?**
Consecutive webcam frames are nearly identical, so a random split puts a near-copy of each test frame in the training set. The model then looks excellent by remembering, not generalising. Holding out an entire session also holds out its room, light and posture, which is what "tomorrow" looks like. The cost is fewer, noisier estimates, so I report per-session accuracy and treat any gap smaller than the spread between sessions as no difference.

**2. Is the ML classifier better than your rules?**
I do not assume it. Both are scored on the same held-out frames, and the report lists the classes where the rules win. Rules are exact on poses they were written for and rotation-invariant by construction; a model can win on messy in-between poses and the `none` class, and can lose on poses it saw little of. If the gap is inside the session-to-session spread I say there is no clear winner. (Fill in your real result here.)

**3. How do you know you are not overfitting to yourself?**
I cannot fully; it is one person and one camera, and I say so. What I do control: hyper-parameters were fixed before looking at data, so nothing was tuned on test sessions; scaling and weights are fitted on training sessions only; and the unknown `none` class plus a confidence threshold stop the model from forcing a named gesture onto every frame. Generalising to other people needs other people's data.

**4. Why export to numpy arrays and not pickle, and why not deploy the random forest?**
A model file is something people download. Unpickling can execute arbitrary code; an `.npz` loaded with `allow_pickle=False` cannot, and a test proves a pickled payload is refused and not run. The trade-off is that only models with a short numpy forward pass can ship: logistic regression and a small MLP. The forest is in the comparison but is not deployable this way. A side benefit is that the app does not need scikit-learn.

**5. What does the classifier cost, and what did you not measure?**
The engine times each stage per frame (capture, MediaPipe, classify, controllers, overlay) and I report mean and p95 with and without ML. I expect MediaPipe to dominate and classification to be a small fraction, but I quote the measured numbers. Not measured: real OS call time (controllers ran against a fake input backend on purpose) and the GUI thread. FPS with a camera is capped by the camera, so I also report headroom.

**Bonus: what went wrong while building it?**
My rotation normalisation had a sign error that only worked for upright hands. Upright unit tests passed. I noticed the evaluation's accuracy under rotation was not flat (70 percent at 90 degrees, 96 percent at 180), which cannot happen with rotation-invariant features, found the sign, and added a test that checks invariance at six angles for every pose (and fails on the old code).
