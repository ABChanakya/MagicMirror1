# MagicMirror gesture recognition

A camera-based gesture classifier (`swipe_left` / `swipe_right` / `null`) for
the MagicMirror project, plus the synthetic-data pipeline built to fix its
weak spot: it generalizes poorly to recording sessions it hasn't seen
before, because the video branch was learning to recognize the room and
lighting instead of the gesture.

This file is the map. Each piece below has its own docs where the detail
actually lives — this just explains what everything is, how it fits
together, and what to do next.

## The core problem, in one number

Trained and evaluated with a random clip-level split, the model looks
excellent (~86% balanced accuracy in cross-validation). Evaluated instead
on an entire recording session held out of training (`train_twostream.py
--split session`, which is the split that matters — see
`gesture_system/train_twostream.py`'s own docstring for why a random split
hides this), the video/fusion branch fell to **22.9%, below chance**. It had
learned "which room is this" instead of "which way did the hand move."

The landmark-only branch (skeleton points, no pixels) never had this
problem, because it never sees the room. So the strategy became: don't fix
the video branch by trying to generate photorealistic training video (four
different approaches at that were all abandoned, see below) — fix it by
generating *appearance-diverse* video from *already-correct* motion, or
just lean on the landmark branch, which doesn't need photorealism at all.

## Layout

```
gesture_system/       real-data pipeline, landmark classifier, KiMoDo synthetic-motion pipeline
camera/                recording + face-anonymization tooling (see camera/README* if present)
unity/GestureMotionPOC/  Unity project that renders synthetic motion in varied rooms/lighting/bodies
docs/environment-build.md  build log for the Unity environments (room by room, decision by decision)
```

## 1. Real-data classifier (`gesture_system/`)

- `preprocess_holistic.py` — MediaPipe Holistic extraction: 56 points (14
  body joints + 2×21 hand points, no face — GDPR), no legs (Holistic
  doesn't see them reliably from a desk-mounted camera). Writes
  `data/landmarks_holistic.npz`.
- `train_twostream.py` — trains and compares three modes on identical
  splits: `landmark` (skeleton only), `video` (frozen-backbone Video Swin),
  `fusion` (both). **Always use `--split session`** to get a real number;
  `--split clip` (the default) is optimistic and hides the exact shortcut
  this project exists to catch.
- Current best, real data only, landmark-only, held-out session:
  **61.5% balanced accuracy** (swipe_left 84%, swipe_right 45%, null 55%).
  swipe_right is the persistently weak class across every experiment below.

## 2. Synthetic motion via KiMoDo (`gesture_system/kimodo_*.py`)

[KiMoDo](https://github.com/nv-tlabs/kimodo) (NVIDIA) is a text-to-3D-motion
diffusion model. It needs the real `kimodo` package (torch, skeleton
forward-kinematics) and runs on the cluster (`kimodo_venv`), not locally —
see `slurm_kimodo_generate.sh` for the working setup (missing-headers
workaround, venv, etc.).

**Why motion, not video:** four separate approaches to generating
photorealistic synthetic *video* (Wan2.2-Animate, LTX-2.5 IC-LoRA,
per-frame SDXL+ControlNet) all landed at 25-30% validator pass rate or
worse, and the per-frame approach was visually rejected outright. The
landmark branch doesn't need pixels, so it doesn't need any of that — it
just needs correct 3D motion, which KiMoDo produces directly.

- `kimodo_to_landmarks.py` — maps KiMoDo's 77-joint SOMA skeleton onto the
  project's 56-point layout and validates wrist trajectories directly in
  skeleton space (`check_trajectory`/`direction_ok`). Two real,
  non-obvious bugs were found and fixed here empirically, not assumed —
  read the docstrings before changing axis handling again:
  - KiMoDo is Y-up; MediaPipe is Y-down. Unflipped, every synthetic clip's
    body shape was vertically mirrored relative to real clips.
  - KiMoDo's raw +X doesn't read as "screen-right" the way a naive prompt
    ("swipe right") would suggest — confirmed by tallying a 20-clip batch
    under both conventions (15% -> 65% pass rate).
- Plain text-to-motion pass rate at real scale (200/class): **52%**
  swipe overall, **73%** null. (A 10-clip sample suggested 65% — small
  samples here are unreliable; trust the 200-clip number.)
- Merging validated plain-prompt clips into training:
  **real-only 61.5% -> 66.9% balanced accuracy** — the best result so far,
  from `data/landmarks_holistic_plus_synth.npz`
  (`merge_synthetic.py --synth-dir gesture_system/`).

### Keyframe-constrained generation (`build_constraint_targets.py`, `kimodo_solve_constraint.py`)

Anchors KiMoDo's output to a *real* clip's own wrist trajectory (start/
mid/end keyframes) instead of hoping a text prompt produces correct
motion, via gradient-descent IK through KiMoDo's own forward-kinematics —
not a hand-derived formula, so it can't get KiMoDo's bone-frame
conventions wrong.

- **Verdict: doesn't beat plain prompting at scale.** Small samples looked
  great (4/4, 3/8) but a proper 116-clip batch came back at 24-61% pass
  and, merged into training, **65.3% balanced accuracy — flat vs. the
  66.9% plain-prompt baseline, not an improvement.** Don't chase this
  further for landmark-branch training data; the infrastructure remains
  useful for the Unity/BVH path below.
- Real bugs found and fixed along the way, useful if this is revisited:
  - `direction_ok()`'s sign-check must be used even when the label is
    already known (from a real source clip) — `train_twostream.py`'s
    flip-augmentation mirrors X and swaps labels 0<->1, so *all* swipe
    data needs one consistent sign convention or flip-augmenting a
    sign-inconsistent clip corrupts the *other* class (measured:
    66.9% -> 53.2%).
  - Position-only IK constraints can put the arm through the torso,
    especially cross-body; `--reg-weight` in `kimodo_solve_constraint.py`
    penalizes deviation from the template's natural pose to discourage
    this.
  - The text prompt still implies a hand ("swipes one hand to the right"
    reads as "the right hand" to the model) even when a *different* hand
    is constrained — `--pin-other-hand` (default on) also constrains the
    inactive hand to its resting position so the prompt can't pull it
    into motion.
- `kimodo_pure_ik_retarget.py` + `kimodo_export_bvh.py` — the
  no-diffusion-at-all option: solves IK against *every* frame of a real
  clip (not just 3-4 keyframes) and exports straight to BVH via KiMoDo's
  own official export function. Used to get real motion into Unity for
  direct comparison against generated clips.

### Depth-estimation experiment (`test_depth_fusion.py`)

MediaPipe's own Z (depth) is noisy — frame-to-frame jumps of 1.4+ units
where X/Y jump under 0.1 on the same clips, which is why constraint
targets exclude real Z entirely and let the template body supply it.
Tested whether sampling a dedicated monocular depth model (Depth Anything
V2, chosen specifically because it's *not* gated like the SMPL-based
alternatives — WHAM, 4D-Humans, HybrIK all require registering for gated
body-model weights, a real setup cost) at MediaPipe's own pixel location
would give a cleaner Z.

**Result: no, at least not this way.** Even after correcting for
scale, Depth Anything's frame-to-frame jitter came out ~55-58% *higher*
than MediaPipe's own — likely because it has no temporal awareness
(each frame processed independently) where MediaPipe Holistic tracks.
Not worth pursuing further without a temporal model
(MotionBERT was the next candidate if this becomes worth revisiting —
takes MediaPipe's already-good 2D points and lifts them to smoothed 3D,
no gating, tiny model — but this is not validated, just researched).

## 3. Unity rendering (`unity/GestureMotionPOC/`)

Full detail: `unity/GestureMotionPOC/README.md` and
`docs/environment-build.md` (a room-by-room build log). Summary:

| Part | State |
|---|---|
| 8 indoor zones, 7 lighting presets each | done, lightmaps baked |
| 10 humanoid characters | done |
| 291 motion clips (KiMoDo -> BVH), clip viewer | done, reviewed |
| Outdoor zones | not built |
| Automated batch renderer (zone x lighting x character x clip -> labeled footage) | **not done** — still an old prefab-based loop, needs replacing |

**This is the actual next milestone for the original problem**: none of
the work above has produced training footage for the video/fusion branch
yet, only the environment/character/motion infrastructure to do it. The
video branch's 22.9% held-out number is exactly as broken as it was on
day one until the batch renderer exists and a retrain is measured against
it.

## Known blocker as of this writing

Cluster SSH access (`cbhaskara@sl-sn02`) is locked out — likely an
automated block from repeated failed logins, possibly needs IT to lift.
Nothing generated so far is at risk (submitted Slurm jobs keep running on
their compute node independently of login-node access), but nothing new
can be generated or validated on the cluster until access returns.

## Next steps, roughly in priority order

1. **Build the Unity batch renderer.** This is the one piece that actually
   turns "we can render a swipe in 8 rooms" into "here is labeled training
   footage" — everything upstream of it is already done.
2. **Retrain the video/fusion branch** on that rendered footage and measure
   `--split session` balanced accuracy against the 22.9% baseline. This is
   the number that actually answers whether the Unity approach worked.
3. Once cluster access is back: merge the `gesture-3class-pipeline` branch's
   Unity work into `main` (see below) if not already done, and consider
   whether the pure-IK-retargeted real clips (no diffusion) are worth
   feeding into Unity alongside the KiMoDo-generated ones.
4. Landmark branch is likely done for now at 66.9% — swipe_right recall
   (45-52% across every variant tried) looks like a ceiling that more
   synthetic *motion* data doesn't move; it may need more/better real
   swipe_right recordings instead.
5. Lower priority: MotionBERT for MediaPipe's Z noise, if the Unity/video
   work ends up needing better real depth.

## Branches

- `main` — this branch. The real-data pipeline, KiMoDo synthetic-motion
  work, and this README.
- `gesture-3class-pipeline` — the Unity environment/rendering work
  (separate machine, diverged from `main` a while back). Needs merging via
  a GitHub pull request (base `main`, compare `gesture-3class-pipeline`) —
  do this on GitHub's web UI rather than locally if local disk is tight;
  the Unity assets are large.
- `mm3-push-clean` — stale initial-import scaffolding, long superseded by
  `main`. Safe to ignore or delete.

## Standing rules for whoever/whatever works on this next

- No emails get sent by an assistant on this project — draft only, always.
- Real, non-anonymized photos exist in `camera/dataset/Chanakya/*.jpg` in
  git history (committed 2026-05-17) — known, intentionally left alone.
- Don't commit generated `.npz` motion/constraint data — see `.gitignore`,
  it's all regenerable and was already 250MB+ once and growing.
