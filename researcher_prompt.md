## Task: pick/validate a pose-guided human-animation model for gesture-data augmentation

### Background

I'm building a hand-gesture recognition system (3 classes: swipe_left, swipe_right,
null) trained on ~930 real recorded clips (~1-2s each, ~30 frames, person
upper-body, facing camera). A prior experiment found the video-based branch of
the classifier scores *below chance* (22.9% balanced accuracy) on a held-out
recording session, because it learns to recognize the session's room/lighting/
clothing instead of the actual hand motion — a classic appearance shortcut.

The fix being attempted: generate synthetic training clips via motion transfer —
take a real clip (motion + pose), swap in a different reference person's
appearance and a different background, keep the original hand motion and its
label. This decorrelates appearance from label, forcing the classifier to
learn the motion itself.

### The problem

Currently using **Wan2.2-Animate-2-14B-Distilled** (via HuggingFace `diffusers`,
`WanAnimate2DistilledModularPipeline`) for this: reference image + driving
video → new video. Built an automated validator (MediaPipe pose extraction +
horizontal displacement direction/magnitude check) rather than relying on
manual inspection.

**Result across a real sample of 911 generated clips: only 25% pass** — the
model frequently either barely moves the hand or moves it in the wrong
direction (swipe_right is the worst-performing class). Accepted clips have
99% pose-detection coverage and travel 0.94±0.81 (arbitrary units), so when
it works it works well — it's specifically unreliable, not uniformly bad.

**Already ruled out as the cause:**
- GPU memory / resolution: tested at 112×112 (too small — pose detection failed
  entirely, 0% pose coverage), 240×320, 384×576, and would-be-native 480×704
  (OOMs on a single 80GB H100 with the model's text encoder resident). The
  240×320/384×576 range gives correct pose tracking (93-100% coverage) with
  ~50-60GB peak VRAM, so there's headroom to spare on an H100.
- Inference steps: this is a *distilled* checkpoint, default/intended
  `num_inference_steps=10` without classifier-free guidance (confirmed in the
  model's own source comments). Raising steps to 20 did not reliably improve
  the pass rate on a small sample.
- Frame extraction / mirroring bugs: checked the frame-extraction code path,
  frames are read in correct temporal order with correct BGR→RGB conversion,
  no flip.

**A relevant constraint:** the driving video has the original person's face
blacked out (a box drawn over it) for privacy/anonymization before it's ever
used as motion input. Pose estimators can partly rely on facial landmarks to
localize a person, so this may itself be degrading the pose/motion signal fed
into whichever model is used — this needs checking regardless of which model
is chosen, since it's a property of the input data, not of Wan-Animate
specifically.

### Hardware / practical constraints

- Access to a university SLURM cluster, up to 4x NVIDIA H100 80GB per job
  (shared with other users; QOS caps at 4 GPUs per user total).
- No root/sudo on cluster nodes — anything needed system-wide (apt packages)
  has to be worked around via user-space installs (e.g. `apt-get download` +
  manual extraction).
- Existing pipeline uses HuggingFace `diffusers`; anything with a native
  `diffusers` pipeline integrates far more easily than a model requiring its
  own separate codebase/repo.
- Reference images (~20 synthetic people × ~16 backgrounds, SDXL-generated)
  already exist and give per-person + per-background variety independent of
  whichever motion-transfer model is used.

### What I need

A recommendation (with reasoning, not just a name) on which current
pose-guided human-image-animation model would most reliably reproduce a
driving video's exact hand-motion direction/magnitude for short simple
single-direction gestures, given:

1. The driving video's face is deliberately obscured (does this model's pose
   extraction depend on facial landmarks, or is it robust to that?).
2. Identity/facial fidelity to the reference image is *not* a priority —
   appearance variety is what's wanted, not identity preservation.
3. Integration cost matters — native `diffusers` support, or at least an
   actively maintained repo with clear inference scripts, is strongly
   preferred over a research-code-only release.
4. Available compute is generous (H100 80GB x up to 4) but not unlimited —
   avoid anything requiring, e.g., 8x A100 to run at all.

Candidates already identified from a first pass (evaluate these, and name
better ones if they exist): **StableAnimator/StableAnimator++**, **MimicMotion**,
**Champ**, and the **full (non-distilled) Wan2.2-Animate-2-14B** as the
cheapest-to-test option since it reuses existing code.

If possible, also flag: does the candidate model's own paper/repo report
per-direction or fine-motion accuracy (as opposed to just overall visual
quality/FID-style metrics), since that's the actual failure mode being fixed
here — most video-generation papers report perceptual quality, not "did the
hand go the correct way."
