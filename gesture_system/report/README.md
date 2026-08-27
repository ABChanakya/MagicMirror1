# Where the Next Clip Counts

**A data-scaling experiment for the MagicMirror gesture classifier.**
Hochschule Landshut · Studienprojekt · 27 August 2026

> 📊 **Interactive version:** [`index.html`](index.html) — open it in a browser
> for the charts with hover readouts. GitHub shows HTML as source, so download
> the file or serve the folder with GitHub Pages.

---

## Finding

**A `null` recording is worth about 27× more per clip than another swipe.**
Both classes still improve the model — but matching what 21 `null` clips gave
took 593 extra swipes.

| | clips added | gain in balanced accuracy | per clip |
|---|---:|---:|---:|
| `null` 8 → 29 | **+21** | +8.7 pts | **0.414 pts** |
| swipes 104 → 697 | **+593** | +9.0 pts | **0.015 pts** |

---

## 1. The question

The classifier recognises three things: a leftward swipe, a rightward swipe, and
`null` — meaning "a hand is visible but this is not a deliberate gesture."
`null` is the class that stops the mirror reacting when someone walks past it or
scratches their face.

Collection so far has been lopsided: **350 left swipes, 347 right swipes, 29
usable `null` clips**. The intuitive response is to record more of everything.
That is expensive, and probably wrong — a class with 350 examples and a class
with 29 are unlikely to benefit equally from one more clip.

So rather than guess, this experiment starves each class in turn. Two curves:
one holds swipes fixed and varies `null`, the other holds `null` fixed and
varies swipes. Comparing how much each curve rises *per clip added* is what
tells us where recording effort pays.

## 2. Method

Every configuration is evaluated with **5-fold cross-validation repeated 3
times** — 15 independent trainings, reported as mean ± 1 standard deviation.

This is not ceremony. With 29 `null` clips, a single 70/15/15 split leaves about
**four** of them in the test set, so `null` recall can only take the values 0%,
25%, 50%, 75% or 100%. Differences between configurations vanish into that
quantisation.

> **Why this mattered.** On a single split, adding class weights looked worth
> **0.6 points** (84.6% → 84.0%) — indistinguishable from noise, and easy to
> dismiss. Under cross-validation the same comparison is worth **2–4 points**
> and holds at every model size tested. The single split was not wrong so much
> as blind.

### Balanced accuracy, not accuracy

With 350/347/29, plain accuracy is close to useless as a target. A model that
predicts `null` **correctly zero times** still scores **95.2%** overall, because
`null` is 4% of the data. Selecting checkpoints on that number actively favours
the model that cannot do `null`'s only job.

Everything below is therefore **balanced accuracy** — the mean of the three
per-class recalls, which weights `null` equally with the two swipe classes.

### Splitting

Folds are drawn over whole recordings, and augmentation is applied inside the
training fold only. An earlier version of this pipeline pre-augmented every clip
to disk and split afterwards, which put augmented copies of validation clips
into training and produced a meaningless 100% validation score.

## 3. Results

### Figure 1 — Balanced accuracy against training-set size

Both panels share a vertical scale, so the difference in slope is the result.

```
        A — varying `null`  (● )              B — varying swipes  (■ )

 90.0% │                                                90.0% │
       │                                                      │
       │                                                      │
 86.3% │                                   ●·······●    86.3% │                                     ······■
       │                                 ··                   │             ·····■···········■······
       │                               ··                     │        ■····
 82.6% │                              ·                 82.6% │      ··
       │                            ··                        │     ·
       │                          ··                          │   ··
 78.9% │       ·····●············●                      78.9% │ ··
       │●······                                               │■
       │                                                      │
 75.2% │                                                75.2% │
       │                                                      │
       └────────────────────────────────────────────          └────────────────────────────────────────────
        8            null clips            29           104          swipe clips              697
```

| Curve | Clips | Balanced acc. | ±1 SD | `null` recall |
|---|---:|---:|---:|---:|
| A · `null` | 8 | 78.1% | ±10.3 | 66.7% |
| A · `null` | 14 | 78.4% | ±12.1 | 64.3% |
| A · `null` | 20 | 78.9% | ±9.1 | 60.0% |
| A · `null` | 25 | 85.9% | ±6.1 | 78.7% |
| A · `null` | **29** | **86.8%** | ±6.6 | 79.3% |
| B · swipes | 104 | 77.8% | ±9.3 | 81.6% |
| B · swipes | 209 | 83.3% | ±7.6 | 79.3% |
| B · swipes | 348 | 85.2% | ±5.5 | 81.6% |
| B · swipes | 522 | 85.2% | ±7.2 | 73.6% |
| B · swipes | **697** | **86.8%** | ±6.6 | 79.3% |

Note that curve A's error bars **shrink** as `null` grows (±10.3 → ±6.6). More
`null` data stabilises the measurement, not just the mean.

### Figure 2 — Class weighting

Four architectures, 25 runs each, isolating inverse-frequency loss weighting.

| Architecture | Params | Weighting | Balanced | Plain acc. | `null` recall |
|---|---:|---|---:|---:|---:|
| tiny | 115K | on | 83.7% | 88.4% | **73%** |
| tiny | 115K | off | 81.3% | 88.8% | 64% |
| small | 642K | on | 85.0% | 89.1% | **76%** |
| small | 642K | off | 82.9% | 88.4% | 70% |
| **wide** | **1.43M** | **on** | **86.3%** | **90.6%** | **77%** |
| wide | 1.43M | off | 82.4% | 89.0% | 68% |
| medium | 3.32M | on | 84.4% | 89.8% | **72%** |
| medium | 3.32M | off | 83.2% | **90.2%** | 67% |

Weighting never costs more than a point of plain accuracy and returns **5 to 9
points** of `null` recall. Note the last row: the largest model without
weighting has the **highest plain accuracy in the entire sweep (90.2%)** and one
of the worst `null` recalls. That is the clearest single illustration of why
plain accuracy was abandoned as a target.

### Figure 3 — What the `null` clips actually contain

Auditing all 35 recorded `null` clips by how many of their 30 frames contain a
detected hand:

| Bucket | Clips | Fate |
|---|---:|---|
| No hand in any frame | 6 | dropped at preprocessing |
| Hand in 1–14 frames | 24 | **gated at inference — the model never runs** |
| Hand in ≥15 frames | **5** | the only clips that test the model |

MediaPipe finds no hand in **75.3%** of `null` frames, against 13–15% for the
swipe classes. Of those 5 useful clips, **one** is held out from training.

## 4. What this means for collection

Both curves rise, so neither class is saturated. What separates them is cost.
Going from 8 to 29 `null` clips bought **8.7 points** for **21 recordings**.
Going from 104 to 697 swipes bought a near-identical **9.0 points** — for **593
recordings**. That is **0.41** points per `null` clip against **0.015** per
swipe clip. An hour spent recording `null` is worth close to a day spent
recording swipes.

There is a sharper finding underneath the counts. Inference already applies a
**hand-presence gate**: if fewer than 15 of 30 frames contain a detected hand,
the window is reported as `null` without the model running at all. That rule is
deterministic and cannot be wrong.

So 30 of the 35 `null` clips train the model on a decision it never gets to
make. The remaining five carry the case that can actually produce a false
trigger: *a hand clearly in frame, moving, but not swiping*. The measured 77%
`null` recall is largely the model learning to recognise an empty room.

> ### Recommendation
>
> Record roughly **100 `null` clips with the hand visible throughout** —
> reaching across the mirror, adjusting hair or glasses, gesturing while
> talking, hovering, drifting diagonally, and starting a swipe then stopping.
> The near-misses are the valuable ones; clips of an empty room add nothing the
> gate does not already handle. More swipes would still help, but at roughly
> 1/27 the rate per clip, so `null` comes first.

Verify the new clips are the right kind with `python audit_null.py` — the
"hand in ≥15 frames" bucket should jump from 5 to most of them.

## 5. Limitations

- All recordings come from **one person, in one room**. Cross-validation
  measures generalisation to unseen *clips*, not unseen *people*. Every number
  here is an upper bound on what a stranger would experience.
- **Curve A is not monotonic** — flat from 8 to 20, then a jump at 25. With
  ±9–12 point error bars at the low end, individual point-to-point differences
  are not significant. Only the endpoint comparison is.
- Curve A's right-hand end is 29 clips. Extrapolating to 100 assumes the trend
  continues; the data establishes that the curve has *not* flattened, not where
  it eventually will.
- Panel B reduces swipes by random subsampling, which preserves the variety of
  the full set. Collecting 100 swipes from scratch would likely be less diverse
  than 100 sampled from 697, so B is a mildly optimistic view of small swipe
  sets.
- `swipe_up` and `swipe_down` were dropped from the class list; two interface
  actions still bound to them are inactive.

## 6. Reproducing this

```bash
cd gesture_system

# landmark cache (~4 MB, no video frames stored)
python preprocess_landmarks.py

# Figure 1 — the two scaling curves
python scaling_study.py --folds 5 --repeats 3 --epochs 80 \
    --d-model 192 --layers 3 --nhead 6

# Figure 2 — architecture x class weighting
python experiments.py --folds 5 --repeats 5 --epochs 100

# Figure 3 — what the null clips contain
python audit_null.py

# encoding regression tests
python test_pipeline.py

# rebuild report/index.html after editing the report
python report/build.py
```

Seeds are fixed, so fold assignments and reported means reproduce exactly.

---

*Landmark-only pipeline · 726 clips · MediaPipe 21-point hand landmarks · RTX 4090*
