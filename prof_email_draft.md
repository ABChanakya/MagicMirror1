Subject: Update — motion-transfer model quality issue, pausing large-scale generation

Hi Professor [NAME],

Quick update on the synthetic data generation part of the gesture project.

**Context:** to break an appearance-shortcut problem we found earlier (the
video model was learning to recognize the recording session/room instead of
the actual hand gesture — 22.9% balanced accuracy, below chance), I've been
generating synthetic training clips with Wan2.2-Animate: take a real recorded
gesture clip, swap in a different person's appearance and background, keep
the hand motion. The idea is to decorrelate appearance from label so the
model is forced to learn the actual motion.

**The issue:** I built an automated validator that checks each generated clip
for the correct pose and direction (rather than relying on eyeballing), and
across a real sample of 911 generated clips, only 25% actually reproduce the
correct motion — the rest either barely move or go the wrong direction
(swipe_right is especially unreliable). I confirmed this isn't a resolution
or GPU-memory issue (tested at several resolutions on the university's H100
nodes with confirmed correct pose tracking on the clips that do pass) — it's
a genuine limitation of this specific model's motion fidelity.

**What I'm doing about it:** rather than run a much larger (~15,000-clip,
~29-hour) generation job on a pipeline with a known 75% failure rate, I'm
pausing that and evaluating alternative models built specifically for
pose-guided human animation (e.g., StableAnimator++, MimicMotion), which are
explicitly designed to solve the pose-alignment problem I'm seeing. I wanted
to flag this before committing more compute time on the cluster, in case you
have a preference on approach or want to discuss before I continue.

Happy to walk through the validator results or the generated samples if
useful.

Best,
Chanakya
