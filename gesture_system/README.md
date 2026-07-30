# Gesture recognition — landmark-only pipeline

Classifies 5 gestures from a webcam: `swipe_left`, `swipe_right`, `swipe_up`,
`swipe_down`, `null`.

## Quick start

```bash
# 1. Extract landmarks from the recorded clips (~4 MB cache, no frames stored)
python preprocess_landmarks.py

# 2. Train (a few minutes on GPU, works on CPU too)
python train_landmark.py

# 3. Run live
python inference.py --model-path checkpoints/landmark_best.pt --landmark-only

# Sanity-check the encoding at any time
python test_pipeline.py
```

Record more data with `../camera/record_batch.py`, then re-run steps 1–2.

## Why landmark-only

Swipe direction is fully determined by where the wrist travels. MediaPipe
already gives us that, so the model only needs to read geometry:

| | Video Swin-B (legacy `train.py`) | Landmark-only (`train_landmark.py`) |
|---|---|---|
| Parameters | 88 M | ~1–5 M |
| Sees | raw pixels | 21 joint coordinates |
| Generalises to new people | poor — keys on appearance | good — appearance is gone |
| Generalises to new lighting | poor | good |
| Disk per clip | ~4.5 MB (30 frames) | ~8 KB |
| Startup | loads Kinetics weights | instant |

Feeding pixels to a model trained on one person's clips means it can key on
that person's skin, sleeves, room and lighting — all of which correlate with
the label in the training set and none of which transfer. Landmarks discard
all of it.

The legacy two-stream code is still present and working if you want to compare,
but it is not the recommended path.

## Landmark encoding — the important part

`_normalise_landmarks()` in [dataset.py](dataset.py) turns a clip into
**(30, 66)** per-frame features:

```
[ 0:63]  hand shape       21 joints (x,y,z), wrist-relative, hand-scale normalised
[63:66]  wrist trajectory  displacement from frame 0, in hand-size units
```

Both streams are needed. Subtracting the wrist per frame — which is what makes
hand *pose* independent of position — also sets the wrist to `(0,0,0)` in every
frame, which **deletes the swipe entirely**. Before this was fixed,
`swipe_left`, `swipe_right` and `swipe_up` encoded to bit-identical tensors, the
landmark branch carried zero direction information, and the model was forced to
read direction off raw pixels instead. That is why it scored ~100% on held-out
clips of the same person and fell apart on anyone else.

The trajectory stream restores the displacement while keeping the invariances
that matter for generalisation:

- **scale-invariant** — divided by hand size, so distance from the camera and
  hand size don't matter
- **position-invariant** — measured relative to frame 0, so where in the frame
  the gesture happens doesn't matter

`test_pipeline.py` asserts all of this, including that the four directions stay
distinguishable. Run it after touching normalisation or augmentation.

### Channel layout is load-bearing

The trajectory sits at indices 63, 64, 65 as contiguous `(x, y, z)`. Because
`63 % 3 == 0`, the `0::3` / `1::3` / `2::3` strides that `horizontal_flip()` and
`rotation()` in [augmentation.py](augmentation.py) use to reach joint x/y/z
components also reach the trajectory's. **Do not reorder those three values** —
a flip that mirrored the hand but not the trajectory would silently produce
mislabelled training data.

## Train/inference consistency

Inference imports `_normalise_landmarks` from `dataset.py` rather than keeping
its own copy. It previously had a duplicate that drifted from the training
version — a mismatch that throws no error and only shows up as inexplicably bad
accuracy. One definition, imported in both places.

## Leak-free splits

`train_landmark.py` splits on unique clip ids *before* augmenting, and augments
in memory on the train split only. Val and test are untouched original clips.

An earlier version pre-augmented every clip to disk and split afterwards, which
put augmented views of validation clips into training — val accuracy read 100%
while the model had effectively been tested on its own training data. The split
now asserts disjointness and fails loudly if it regresses.

Reported `TEST acc` is measured once, on a split used for neither training nor
model selection.

## Honest expectations

Accuracy on held-out clips **of people already in the training set** tells you
little about how it behaves for a stranger. The real benchmark is someone whose
clips were never recorded. Adding people to the dataset moves that number far
more than any architecture change — the model is not the bottleneck, data
diversity is.
