# Running Wan Animate motion transfer on a university GPU server

The goal is motion control: a **reference image** (who the person looks like) plus
a **driving video** (your recorded gesture), producing the reference character
performing your exact motion. Wan2.2-Animate does this natively — its API is
`image=` + `driving_video=`.

It does not run on the 24 GB RTX 4090. This documents what does.

---

## Why the 4090 failed

| configuration | weights | headroom for activations | result |
|---|---:|---:|---|
| bf16 | 32.8 GB | — | exceeds 24 GB before loading |
| int8 (bitsandbytes) | 18.9 GB | ~5 GB | loads, **OOM during attention** even at 384×320 |
| int8 + CPU offload | 18.9 GB | — | **hangs** — int8 weights don't move between devices |

The transformer is **16.4B parameters**. Quantisation shrinks weights but not the
activation memory that video attention needs, which is what actually ran out.

## What to ask the university for

**Minimum**

- 1× GPU with **≥ 48 GB VRAM** — A100 40 GB is marginal, A6000 48 GB works
- 60 GB disk for model weights
- CUDA 12.x

**Comfortable** (what to ask for if there's a choice)

- 1× **A100 80 GB** or **H100 80 GB**
- bf16, no quantisation, full 704×480 at 81-frame segments
- ~60 s per clip, so ~900 clips is roughly 15 GPU-hours

Ask for an **interactive session** rather than pure batch for the first run —
the first attempt usually needs a few adjustments, and a 20-minute queue per
iteration makes that painful.

Slurm example:

```bash
srun --partition=gpu --gres=gpu:a100:1 --mem=64G --cpus-per-task=8 \
     --time=04:00:00 --pty bash
```

## Setup on the server

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install diffusers accelerate transformers imageio imageio-ffmpeg \
            opencv-python mediapipe pyyaml

# weights go somewhere with space — NOT your home quota
export HF_HOME=/scratch/$USER/hf
python -c "from huggingface_hub import snapshot_download as d; \
           d('Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers', max_workers=8)"
```

Copy across: `animate_variants.py`, `validate_generated.py`,
`preprocess_holistic.py`, `dataset.py`, `config.yaml`, and the clips in
`camera/gesture_training_data/`.

## Running it

```bash
export GEN_OUT=/scratch/$USER/animated

# 1. sanity check — reference is the clip's own first frame.
#    If the output reproduces your gesture, motion transfer works.
python animate_variants.py --self-test --limit 3 --bits 0 \
       --height 480 --width 704 --frames 81

# 2. real run — reference images of different people
python animate_variants.py --refs refs/ --limit 50 --bits 0 \
       --height 480 --width 704 --frames 81

# 3. NEVER skip this
python validate_generated.py --dir $GEN_OUT --by-subdir --move-rejects
```

`--bits 0` means no quantisation, which is the point of using the server.

## The missing input: reference images

Every attempt so far used your own first frame, so even a success would render
*you*. To vary appearance you need stills of different people, matching your
clips' framing: upper body, facing camera, plain background, arms visible.

Either generate them (SDXL or Flux, both fit on the 4090 locally — no server
needed for this part) or use photos. Aim for variation in body type, skin tone,
clothing and lighting, since those are the attributes the video branch currently
keys on.

## Validate before training on any of it

`validate_generated.py` runs MediaPipe Holistic over every generated clip and
checks the pose is detectable, the wrist actually moves, and the sweep runs in
the labelled direction. It was verified against real clips and caught all six
deliberately mislabelled ones it was given.

This is not optional. LTX image-to-video passed **0 of 3** — one clip generated
as `null` came back containing a large sweep. Training on unvalidated generated
data would poison the labels.

## Expected honest outcome

Generated data breaks the appearance-to-session shortcut, which is worth having.
It does **not** add motion diversity — 900 clips re-rendered 10 ways is still 900
distinct motions. And strong classical appearance augmentation, which attacks the
same shortcut, moved the video branch only 22.9% → 25.4% (chance is 33%).

So the experiment worth running on the server is specifically: *does the video
branch clear chance on a held-out session once appearance no longer predicts the
label?* That is a clean, publishable result either way.
