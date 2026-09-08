#!/usr/bin/env python3
"""
generate_variants.py — synthesise appearance variants of existing gesture clips.

Goal: break the appearance-to-session shortcut. The video branch scores 22.9%
balanced on a held-out recording session — below the 33% chance line — because
session 20260827 is 200 `null` clips and nothing else, so "which day is this"
predicts the label better than the gesture does. Re-rendering the same motions
with different people, clothing and lighting removes that correlation.

Two backends, and the choice matters more than the model does:

  v2v    video-to-video. The whole source clip conditions the generation, so the
         motion carries through while the prompt changes the person, clothing,
         background and lighting. `denoise_strength` is the trade-off: too low
         and appearance barely changes, too high and the gesture drifts away
         from its label. Sweep it and let the validator pick.

  i2v    image-to-video from the first frame only. Measured 0/3 on a trial —
         the model gets one frame and a text prompt, with no channel through
         which the gesture could be communicated, so it improvises. A clip asked
         for as `null` came back containing a large sweep. Kept for reference,
         not recommended.

  pose   pose-conditioned generation driven by the 56-point skeleton already
         extracted for every clip. The motion comes from real data, so the label
         is correct by construction. This is the right mode for training data;
         i2v is a fallback when no pose-conditioned checkpoint is available.

Nothing generated here enters training directly. The pipeline is:

    generate -> validate_generated.py -> merge only what passes

Usage
-----
python generate_variants.py --check                       # environment only
python generate_variants.py --backend i2v --per-clip 2 --limit 20 --dry-run
python generate_variants.py --backend i2v --per-clip 2 --limit 20
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# Appearance prompts. Deliberately varied across body type, clothing, skin tone
# and lighting — those are the attributes the video branch is currently keying
# on, so they are exactly what must stop predicting the label.
APPEARANCES = [
    "a woman in a red sweater, bright daylight from a window",
    "an older man in a grey shirt, warm indoor lamp light",
    "a young man in a black hoodie, dim evening light",
    "a woman with dark skin in a yellow blouse, neutral office lighting",
    "a man with light skin in a blue striped shirt, overcast daylight",
    "a teenager in a green t-shirt, harsh overhead light",
    "a woman in a white blouse in front of a plain wall, soft light",
    "a man in a brown jacket, cool fluorescent lighting",
]

MIN_FREE_GB = 40


def check_env(verbose=True):
    ok = True
    free_gb = shutil.disk_usage('/').free / 1e9
    hf = os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))
    hf_free = shutil.disk_usage(Path(hf).parent if Path(hf).parent.exists()
                                else '/').free / 1e9

    def line(label, good, detail):
        nonlocal ok
        if not good:
            ok = False
        if verbose:
            print(f"  [{'OK ' if good else 'FAIL'}] {label:26} {detail}")

    # Weights live under HF_HOME (the HDD); / only needs room for temp files
    # during decode/encode, which is a few hundred MB at these clip sizes.
    line("disk on / (temp only)", free_gb > 1.5, f"{free_gb:.1f} GB free")
    line("HF_HOME", hf_free > MIN_FREE_GB,
         f"{hf} ({hf_free:.1f} GB free, need ~{MIN_FREE_GB})")

    try:
        import torch
        line("torch + CUDA", torch.cuda.is_available(),
             f"{torch.cuda.get_device_name(0)} "
             f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB"
             if torch.cuda.is_available() else "no CUDA")
    except Exception as e:
        line("torch + CUDA", False, str(e)[:50])

    try:
        import diffusers
        line("diffusers", True, diffusers.__version__)
    except Exception:
        line("diffusers", False, "pip install diffusers accelerate transformers")

    if verbose and not ok:
        print("\n  Fix before generating:")
        if hf_free <= MIN_FREE_GB:
            print("    sudo mkdir -p /scratch/bhaskara && \\")
            print("      sudo chown bhaskara:bhaskara /scratch/bhaskara")
            print("    export HF_HOME=/scratch/bhaskara/hf")
    return ok


def pick_sources(cfg, per_class, seed=0):
    """
    Choose real clips to re-render.

    Weighted toward `null`, because that is the class whose appearance is most
    entangled with a single session — 200 of its 235 clips come from one sitting.
    """
    raw = Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]
    rng = random.Random(seed)
    out = []
    for c in classes:
        vids = sorted((raw / c).glob('*.mp4'))
        if not vids:
            continue
        n = per_class.get(c, per_class.get('default', 10))
        # spread across runs so variants are not all from one sitting
        by_run = {}
        for v in vids:
            by_run.setdefault(v.stem.split('_')[-2] if '_' in v.stem else '?', []).append(v)
        picks, runs = [], list(by_run)
        rng.shuffle(runs)
        i = 0
        while len(picks) < min(n, len(vids)):
            r = runs[i % len(runs)]
            if by_run[r]:
                picks.append(by_run[r].pop(rng.randrange(len(by_run[r]))))
            i += 1
            if all(not v for v in by_run.values()):
                break
        out += [(p, c) for p in picks]
    return out


def load_v2v_pipeline(model_id, device='cuda'):
    """Video-to-video: the source clip conditions generation, so motion survives."""
    import torch
    import diffusers
    if 'ltx' in model_id.lower() and hasattr(diffusers, 'LTXConditionPipeline'):
        cls = diffusers.LTXConditionPipeline
    elif hasattr(diffusers, 'CogVideoXVideoToVideoPipeline') and 'cog' in model_id.lower():
        cls = diffusers.CogVideoXVideoToVideoPipeline
    else:
        cls = diffusers.LTXConditionPipeline
    print(f"[gen] using {cls.__name__} (video-to-video)")
    pipe = cls.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)

    # LTXConditionPipeline builds its own timestep schedule, but the flow-match
    # scheduler still insists on a `mu` value while dynamic shifting is enabled
    # and raises before the first step. Turning it off is what makes the
    # video-conditioned path usable at all.
    try:
        if pipe.scheduler.config.get("use_dynamic_shifting"):
            pipe.scheduler.register_to_config(use_dynamic_shifting=False)
            print("[gen] disabled scheduler dynamic shifting (needs `mu` otherwise)")
    except Exception:
        pass

    # memory helpers live on the VAE for this pipeline, not the pipeline itself
    for obj, fn in ((getattr(pipe, 'vae', None), 'enable_tiling'),
                    (getattr(pipe, 'vae', None), 'enable_slicing'),
                    (pipe, 'enable_model_cpu_offload'),
                    (pipe, 'enable_attention_slicing')):
        if obj is not None and hasattr(obj, fn):
            try:
                getattr(obj, fn)()
            except Exception:
                pass
    return pipe


def load_i2v_pipeline(model_id, device='cuda'):
    """
    Load an image-to-video pipeline.

    DiffusionPipeline.from_pretrained resolves LTX-Video to LTXPipeline, which is
    text-to-video and rejects `image=`. The i2v variant is a separate class, so
    pick it explicitly by model family rather than relying on auto-resolution.
    """
    import torch
    import diffusers

    families = [
        ('ltx', 'LTXImageToVideoPipeline'),
        ('wan', 'WanImageToVideoPipeline'),
        ('cogvideox', 'CogVideoXImageToVideoPipeline'),
        ('hunyuan', 'HunyuanVideoImageToVideoPipeline'),
        ('skyreels', 'SkyReelsV2ImageToVideoPipeline'),
    ]
    lower = model_id.lower()
    cls = None
    for key, name in families:
        if key in lower and hasattr(diffusers, name):
            cls = getattr(diffusers, name)
            print(f"[gen] using {name}")
            break
    if cls is None:
        cls = diffusers.AutoPipelineForImage2Image if hasattr(
            diffusers, 'AutoPipelineForImage2Image') else diffusers.DiffusionPipeline
        print(f"[gen] no i2v class matched '{model_id}', falling back to {cls.__name__}")

    print(f"[gen] loading {model_id} — first run downloads several GB")
    pipe = cls.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)
    # 24 GB is tight for a video model; trade speed for memory
    for fn in ('enable_vae_slicing', 'enable_vae_tiling',
               'enable_model_cpu_offload', 'enable_attention_slicing'):
        if hasattr(pipe, fn):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass
    return pipe


def run_v2v(args, sources, outdir):
    """
    Re-render each clip with a new appearance while keeping its motion.

    Strength is the whole ballgame. The validator measures whether the gesture
    survived, so a sweep turns "pick a number" into a measurement: generate the
    same clips at several strengths, validate each, and keep the highest
    strength that still passes.
    """
    import imageio
    import numpy as np
    import torch
    from PIL import Image

    from dataset import _extract_frames

    pipe = load_v2v_pipeline(args.model)
    rng = random.Random(args.seed)
    strengths = ([float(x) for x in args.strength_sweep.split(',')]
                 if args.strength_sweep else [args.strength])
    manifest, made = [], 0
    total = len(sources) * args.per_clip * len(strengths)

    for src, cls in sources:
        frames = _extract_frames(str(src), num_frames=args.frames)
        if frames is None:
            continue
        n_frames = max(9, round((args.frames - 1) / 8) * 8 + 1)
        side = max(64, round(args.size / 32) * 32)
        vid_in = [Image.fromarray(f).resize((side, side)) for f in frames]
        while len(vid_in) < n_frames:
            vid_in.append(vid_in[-1])
        vid_in = vid_in[:n_frames]

        for st in strengths:
            for k in range(args.per_clip):
                look = rng.choice(APPEARANCES)
                prompt = (f"{look}, facing the camera, upper body visible, "
                          f"steady camera, plain background")
                sub = f"s{int(st*100):02d}" if len(strengths) > 1 else ""
                dest = outdir / sub / cls if sub else outdir / cls
                dest.mkdir(parents=True, exist_ok=True)
                name = f"gen_{src.stem}_v{k:02d}.mp4"
                try:
                    out = pipe(video=vid_in, prompt=prompt,
                               negative_prompt="blurry, distorted hands, extra limbs",
                               num_frames=n_frames, height=side, width=side,
                               denoise_strength=st,
                               num_inference_steps=args.steps,
                               generator=torch.Generator('cuda').manual_seed(
                                   rng.randrange(1 << 30)))
                    imageio.mimsave(str(dest / name), out.frames[0], fps=15)
                    manifest.append({'file': str(dest / name), 'source': str(src),
                                     'label': cls, 'appearance': look,
                                     'strength': st})
                    made += 1
                    print(f"  [{made}/{total}] s={st:.2f} {cls:12} {name}")
                except Exception as e:
                    print(f"  FAILED s={st:.2f} {name}: {str(e)[:90]}")

    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"\ngenerated {made} clips -> {outdir}")
    if len(strengths) > 1:
        print("\nValidate each strength and keep the highest one that survives:")
        for st in strengths:
            print(f"  python validate_generated.py --dir {outdir}/s{int(st*100):02d} --by-subdir")
    else:
        print(f"\n  python validate_generated.py --dir {outdir} --by-subdir --move-rejects")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--backend', default='v2v', choices=['v2v', 'i2v', 'pose'])
    ap.add_argument('--strength', type=float, default=0.5,
                    help='denoise strength for v2v: low keeps the original motion '
                         'and changes little, high restyles more but drifts off '
                         'the gesture. Sweep it with --strength-sweep.')
    ap.add_argument('--strength-sweep', default='',
                    help='comma-separated strengths to compare, e.g. 0.3,0.5,0.7')
    ap.add_argument('--model', default='Lightricks/LTX-Video',
                    help='HF id; LTX is the practical choice on a single 24 GB card')
    ap.add_argument('--out', default='generated')
    ap.add_argument('--per-clip', type=int, default=2)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--null-weight', type=float, default=2.0,
                    help='how much more `null` to generate — it is the class most '
                         'tied to a single recording session')
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    print("Environment")
    ok = check_env()
    if args.check:
        return 0 if ok else 1

    cfg = yaml.safe_load(open(args.config))
    classes = [str(c) for c in cfg['data']['classes']]
    base = 12
    per_class = {c: int(base * (args.null_weight if c == 'null' else 1))
                 for c in classes}
    sources = pick_sources(cfg, per_class, args.seed)
    if args.limit:
        # Round-robin across classes. Slicing the flat list instead would take
        # `limit` clips of whichever class came first and none of the others.
        by_cls = {}
        for p, c in sources:
            by_cls.setdefault(c, []).append(p)
        picked, i = [], 0
        while len(picked) < args.limit and any(by_cls.values()):
            c = classes[i % len(classes)]
            if by_cls.get(c):
                picked.append((by_cls[c].pop(0), c))
            i += 1
        sources = picked

    total = len(sources) * args.per_clip
    print(f"\nPlan")
    print(f"  backend        {args.backend}")
    print(f"  source clips   {len(sources)}  " +
          "  ".join(f"{c}={sum(1 for _, k in sources if k == c)}" for c in classes))
    print(f"  variants each  {args.per_clip}")
    print(f"  total to make  {total}")
    print(f"  output         {args.out}/<class>/")

    if args.backend == 'i2v':
        print("\n  NOTE: i2v invents motion. Expect a substantial reject rate from")
        print("  validate_generated.py — that is the gate working, not a failure.")

    if args.dry_run:
        print("\n[dry-run] nothing generated. Sources that would be used:")
        for p, c in sources[:10]:
            print(f"    {c:12} {p.name}")
        if len(sources) > 10:
            print(f"    ... and {len(sources)-10} more")
        return 0

    if not ok:
        print("\nEnvironment not ready — see the fixes above. Nothing generated.")
        return 1

    if args.backend == 'pose':
        print("\nPose-conditioned generation needs a checkpoint that accepts a pose")
        print("sequence (AnimateAnyone / MagicAnimate / Champ lineage). Point --model")
        print("at one; the skeletons are already in data/landmarks_holistic.npz.")
        return 1

    import imageio
    import torch
    from dataset import _extract_frames

    outdir = Path(args.out)
    if args.backend == 'v2v':
        return run_v2v(args, sources, outdir)
    pipe = load_i2v_pipeline(args.model)
    rng = random.Random(args.seed)
    manifest = []
    made = 0

    for src, cls in sources:
        frames = _extract_frames(str(src), num_frames=args.frames)
        if frames is None:
            continue
        from PIL import Image
        first = Image.fromarray(frames[0])          # pipelines expect PIL, not ndarray

        # LTX requires num_frames = 8n+1 and side lengths divisible by 32.
        n_frames = max(9, round((args.frames - 1) / 8) * 8 + 1)
        side = max(64, round(args.size / 32) * 32)

        for k in range(args.per_clip):
            look = rng.choice(APPEARANCES)
            prompt = (f"{look}, facing the camera, performing a clear hand gesture, "
                      f"upper body visible, steady camera")
            dest = outdir / cls
            dest.mkdir(parents=True, exist_ok=True)
            name = f"gen_{src.stem}_v{k:02d}.mp4"
            try:
                result = pipe(prompt=prompt, image=first,
                              num_frames=n_frames,
                              num_inference_steps=args.steps,
                              height=side, width=side,
                              generator=torch.Generator('cuda').manual_seed(
                                  rng.randrange(1 << 30)))
                vid = result.frames[0]
                imageio.mimsave(str(dest / name), vid, fps=15)
                manifest.append({'file': str(dest / name), 'source': str(src),
                                 'label': cls, 'appearance': look})
                made += 1
                print(f"  [{made}/{total}] {cls:12} {name}")
            except Exception as e:
                print(f"  FAILED {name}: {str(e)[:90]}")

    Path(outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"\ngenerated {made} clips -> {outdir}")
    print("\nNext, and do not skip it:")
    print(f"  python validate_generated.py --dir {outdir} --by-subdir --move-rejects")
    print("  Only clips that pass carry the label they claim.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
