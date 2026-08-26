# Handoff — gesture pipeline, Mac → Linux training box

Read this first if you are Claude on `k017-ws22`. It covers what changed, what
is already on disk, and the one thing that will break if you skip it.

Written 2026-08-26. Data was collected on the MacBook Air; training happens here.

---

## 1. State of the data

Already transferred by rsync — **do not re-sync, it is complete and verified**:

```
~/Desktop/Magicmirror3/camera/gesture_training_data/
    swipe_left     350 clips  (+ 350 .json sidecars)
    swipe_right    347 clips  (+ 347 .json sidecars)
    null            35 clips  (+  35 .json sidecars)
    swipe_up         0
    swipe_down       0
                   732 clips / 462 MB
```

Counts and total size match the Mac exactly, and every `.mp4` has its sidecar.

An **older, unrelated dataset** sits beside it at `camera/old_data/` (558 clips,
190 MB, May–Jul 2026). It is deliberately kept out of the way:

- different naming (`01.mp4` … `99.mp4`), no sidecars, unknown provenance
- it contains `swipe_up`/`swipe_down`, classes we have since dropped
- **its swipe-direction convention is unverified.** Our clips are labelled from
  the *camera's* point of view. If `old_data` used the user's point of view,
  merging it would put contradictory labels inside `swipe_left`/`swipe_right`
  and neither class would be learnable.

Do not merge `old_data` into training without checking that convention first.
Chanakya wants to revisit it later, not now.

## 2. The model is 3-class now, not 5

`swipe_up` and `swipe_down` were dropped — no data was ever collected for them,
and the clips that were sitting in those folders turned out to be mislabelled
swipes that have since been moved to their correct classes.

`gesture_system/config.yaml` now reads:

```yaml
classes: [swipe_left, swipe_right, "null"]
num_classes: 3
```

### The bug this branch fixes — do not undo it

`augmentation.py` used to hard-code the class indices:

```python
SWIPE_LEFT = 0;  SWIPE_RIGHT = 1;  SWIPE_UP = 2;  SWIPE_DOWN = 3;  VOID = 4
```

With a 3-class list, `null` moves to index 2 while `VOID` still said 4. The
horizontal-flip augmentation looks up a clip's new label in a map keyed by these
constants, so every mirrored `null` clip would be relabelled as whatever sits at
index 2 — silently, with no error, poisoning training.

The constants are now derived from `config.yaml` at import, and the flip map is
built **by class name**, so it stays correct whatever the class order:

```python
CLASSES = _class_order()            # read from config.yaml
VOID     = _index_of('null')        # 2 in a 3-class config, 4 in a 5-class one
_FLIP_LABEL_MAP = {_index_of(n): _index_of(_FLIP_NAME_MAP.get(n, n)) for n in CLASSES}
```

Classes absent from the config resolve to `-1` rather than colliding with a real
index. If you ever add `swipe_up`/`swipe_down` back, edit `config.yaml` only —
`augmentation.py` follows automatically.

## 3. Getting this code

The Linux checkout was on `main` at `25edc2b`, which predates all of the above.

```bash
cd ~/Desktop/Magicmirror3
git fetch origin
git checkout gesture-3class-pipeline
```

Verify you have the right code before running anything:

```bash
grep -n "classes:\|num_classes:" gesture_system/config.yaml
# expect: [swipe_left, swipe_right, "null"]  and  num_classes: 3
grep -n "_class_order" gesture_system/augmentation.py
# expect a match — if this prints nothing you are on the old code
```

## 4. Running the pipeline

```bash
cd gesture_system
python preprocess_landmarks.py     # -> data/landmarks.npz  (~4 MB, no frames stored)
python train_landmark.py           # --epochs 60 --lr 3e-4, or --smoke-test
```

`train_landmark.py` takes `num_classes` from `len(classes)`, so it follows the
config. `inference.py` reads `model.num_classes` directly — that is why both
fields in `config.yaml` had to change together.

`python test_pipeline.py` passes 24/24 on this branch. Run it if you touch the
augmentation or encoding path.

## 5. Known state / open items for tomorrow

- **Topic for tomorrow is training architecture.** Chanakya is heading home;
  nothing is expected to be trained tonight.
- **`null` is thin — 35 clips against a target of 100.** With only two real
  gesture classes, `null` is what stops the model firing on incidental hand
  movement. It is the weakest part of the dataset and the most likely source of
  false positives at inference.
- **Class imbalance:** 350 / 347 / 35. Worth weighting the loss or resampling
  rather than training on it raw.
- **Planned data expansion:** generate variants of existing clips with local
  generative video models (varying clothing and appearance) so the model keys on
  hand motion rather than the person. Discussed with the supervisor; not built.
- **Disk here is tight:** 5.3 GB free on `/`. The landmark-only pipeline is fine
  (~4 MB cache). The legacy two-stream pipeline in `train.py` stores 30 frames
  per clip and needs tens of GB — it will not fit. Use `train_landmark.py`.
- The MagicMirror UI still binds `pageHomeGesture: "swipe_down"` and
  `interactGesture: "swipe_up"` in `MMM-CameraBridge.js`. Both gestures are now
  unpredictable by the model, so those two actions are dead until rebound.
  Left alone deliberately — it is a UX decision, not a code fix.

## 6. Collecting more data (on the Mac, for reference)

The collector was simplified: one take per prompt, every take kept, no retries.

```bash
cd camera
./collect.sh cameras          # pick the webcam by eye; indices shuffle on replug
./collect.sh                  # ENTER in the video window starts each take
```

If the webcam drops off USB mid-session it now aborts with a clear message
instead of writing header-only 257-byte files and retrying them hundreds of
times.
