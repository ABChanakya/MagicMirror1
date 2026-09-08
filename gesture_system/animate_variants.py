#!/usr/bin/env python3
"""
animate_variants.py — re-render gesture clips as a different person, keeping the motion.

Uses Wan2.2-Animate-2-14B-Distilled: it takes a reference image (who the person
should look like) and a driving video (what they should do), and animates the
reference character with the driving video's motion. Unlike image-to-video, the
gesture is not invented — it is transferred from the real recording, so the label
stays correct.

Fitting it on 24 GB:
  16.4B params, bf16 on disk  -> 32.8 GB, does not fit
  int8 via bitsandbytes       -> 18.9 GB measured, fits with room for activations

The text encoder (UMT5, 11.4 GB) cannot also sit on the card, so model CPU
offload moves components on and off as they are used.

Usage
-----
python animate_variants.py --self-test          # reference = the clip's own frame
python animate_variants.py --refs refs/ --limit 6
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

MID = "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"

# The reference image carries appearance, so the prompt is spent on scene and
# motion quality instead. Naming finger/hand integrity matters because the
# validator reads the wrist trajectory: mangled hands mean no pose, and a clip
# that cannot be tracked cannot be used regardless of how it looks.
PROMPT = (
    "A person standing facing the camera in a well-lit indoor room, upper body "
    "and both arms clearly visible, raising one hand and sweeping it smoothly "
    "across the frame in a single deliberate motion, natural human proportions, "
    "five fingers per hand, sharp focus on the hands, static locked-off camera, "
    "plain uncluttered background, realistic photographic lighting"
)
NEGATIVE = (
    "blurry, low quality, distorted hands, extra fingers, missing fingers, "
    "extra limbs, deformed anatomy, warped face, morphing body, jittery motion, "
    "camera shake, zoom, pan, cropped arms, multiple people, text, watermark"
)


def _compute_crop_box(frames, holistic, pad=0.35):
    """
    Tight normalised (x0,y0,x1,y1) bounding box around body+hand landmarks,
    unioned across every frame in the clip — a single stable box for the
    whole clip, not a per-frame one, since a moving crop would look like
    camera pan and add motion the model has to untangle from the actual
    gesture. Every frame, not just first/middle/last: the swipe's widest
    hand excursion happens mid-motion, not at those three points, and an
    earlier version that only sampled those three cut the hand out of frame
    at the swipe's actual peak. Falls back to the full frame if nothing is
    detected (e.g. a heavily-masked clip).

    The point: a swipe that only occupies a small fraction of a wide shot
    gives the model a weak, easily-lost spatial signal. Cropping tighter
    around the person makes the same physical hand travel cover more of the
    frame, which is a stronger, less ambiguous conditioning signal.
    """
    xs, ys = [], []
    for i in range(len(frames)):
        res = holistic.process(frames[i])
        for group in (res.pose_landmarks, res.left_hand_landmarks, res.right_hand_landmarks):
            if group:
                for lm in group.landmark:
                    xs.append(lm.x)
                    ys.append(lm.y)
    if not xs:
        return 0.0, 0.0, 1.0, 1.0
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    return (max(0.0, x0 - w * pad), max(0.0, y0 - h * pad),
            min(1.0, x1 + w * pad), min(1.0, y1 + h * pad))


def _encode_prompt_once(pipe):
    """
    Run the text encoder once and free it.

    Returns a dict of embeddings to pass straight to the denoiser, or {} if the
    encoder is not exposed in a way we can drive — in which case the caller
    falls back to passing raw prompt strings.
    """
    import gc

    import torch
    te = getattr(pipe, 'text_encoder', None)
    tok = getattr(pipe, 'tokenizer', None)
    if te is None or tok is None:
        return {}
    try:
        te.to('cuda')
        out = {}
        for key, text in (('prompt_embeds', PROMPT),
                          ('negative_prompt_embeds', NEGATIVE)):
            ids = tok([text], padding='max_length', max_length=512,
                      truncation=True, return_tensors='pt')
            with torch.no_grad():
                e = te(ids.input_ids.to('cuda'),
                       attention_mask=ids.attention_mask.to('cuda'))[0]
            out[key] = e.to(torch.bfloat16)
        print(f"[wan] prompt encoded {tuple(out['prompt_embeds'].shape)}, freeing UMT5")
        # Move to CPU rather than pipe.update_components(text_encoder=None): the
        # modular pipeline reads pipe.text_encoder.dtype elsewhere regardless of
        # whether embeds are used, and a None component broke that lookup. CPU
        # residency still frees the 11.4 GB of GPU memory that mattered here.
        te.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        return out
    except Exception as e:
        print(f"[wan] prompt pre-encode failed ({str(e)[:70]}); using raw prompts")
        try:
            getattr(pipe, 'text_encoder').to('cpu')
        except Exception:
            pass
        return {}


def load_pipeline(steps_hint=10, bits=0):
    import torch
    from diffusers import (WanAnimate2DistilledModularPipeline,
                           WanAnimate2Transformer3DModel)

    # Weight size is not the whole story. int8 fits the weights at 18.9 GB but
    # leaves only ~5 GB for activations, and attention over a video sequence
    # needs more than that — hence OOM at 384x320 on a 24 GB card. 4-bit NF4
    # roughly halves the weights again to ~8 GB, trading fidelity for headroom.
    # bits=0 skips quantization_config entirely: bf16, full fidelity, for a
    # card with enough VRAM to not need this trade-off (>=48 GB; an H100's
    # 80 GB holds the 32.8 GB transformer with room to spare).
    #
    # BitsAndBytesConfig is imported lazily, inside the branches that actually
    # quantize. Importing it unconditionally drags in bitsandbytes, which runs
    # a torch.compile at import time — multiple minutes the first time, for
    # nothing, on the bits=0 path that never touches it.
    if bits == 0:
        print("[wan] loading transformer in bf16, no quantisation (32.8 GB)")
        tr = WanAnimate2Transformer3DModel.from_pretrained(
            MID, subfolder="transformer", torch_dtype=torch.bfloat16)
    else:
        from diffusers import BitsAndBytesConfig
        if bits == 4:
            qcfg = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True)
            print("[wan] quantising transformer to 4-bit NF4 (~8 GB, leaves room for activations)")
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
            print("[wan] quantising transformer to int8 (~18.9 GB, tight)")
        tr = WanAnimate2Transformer3DModel.from_pretrained(
            MID, subfolder="transformer",
            quantization_config=qcfg,
            torch_dtype=torch.bfloat16)

    # Modular pipelines register component *specs* on from_pretrained and leave
    # the components themselves unloaded — the earlier failure was the text
    # encoder still being None at call time. load_components() materialises
    # them; the already-quantised transformer is then swapped in.
    pipe = WanAnimate2DistilledModularPipeline.from_pretrained(MID)
    missing = list(getattr(pipe, 'null_component_names', []) or [])
    if missing:
        print(f"[wan] loading components: {missing}")
        pipe.load_components(names=[m for m in missing if m != 'transformer'],
                             torch_dtype=torch.bfloat16)
    pipe.update_components(transformer=tr)

    # This modular pipeline always runs the raw prompt through the text encoder
    # itself — passing prompt_embeds does not bypass that (unlike classic
    # diffusers pipelines), and prompt=None fails its required-input check. So
    # the text encoder has to stay resident and loaded; no pre-encode/offload
    # trick is possible here. VRAM headroom for attention comes from
    # height/width/frames instead (see the sbatch script).
    embeds = {}

    for name in ('transformer', 'vae', 'image_encoder', 'text_encoder'):
        comp = getattr(pipe, name, None)
        if comp is not None and hasattr(comp, 'to'):
            try:
                comp.to('cuda')
            except Exception:
                pass
    free, total = torch.cuda.mem_get_info()
    print(f"[wan] VRAM in use {(total-free)/1e9:.1f} / {total/1e9:.1f} GB")
    still = list(getattr(pipe, 'null_component_names', []) or [])
    if still:
        print(f"[wan] WARNING still unloaded: {still}")

    # No CPU offload. bitsandbytes int8 weights cannot be moved between devices
    # cleanly, and combining the two hung the first attempt: 32 minutes at 4% GPU
    # and 0% CPU, stalled rather than slow. At 18.9 GB the transformer fits on
    # the card outright, so nothing needs to move — the text encoder is dealt
    # with separately by encoding prompts before the transformer is resident.
    if hasattr(pipe, 'vae') and hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    return pipe, embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--refs', help='directory of reference person images')
    ap.add_argument('--self-test', action='store_true',
                    help="reference = the clip's own first frame; if the output "
                         "reproduces the original gesture the pipeline is sound")
    ap.add_argument('--out', default=os.environ.get('GEN_OUT',
                                                    '/media/bhaskara/Volume/animated'),
                    help='override with $GEN_OUT so this runs unchanged on a server')
    ap.add_argument('--limit', type=int, default=3,
                    help='source clips PER CLASS (not a total cap)')
    ap.add_argument('--per-clip', type=int, default=1,
                    help='how many reference images (variants) to render per '
                         'source clip; ignored with --self-test')
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--width', type=int, default=704)
    ap.add_argument('--bits', type=int, default=0, choices=[0, 4, 8],
                    help='0 = bf16, needs >=48 GB (use on the server); '
                         '4/8 = quantised attempts for a 24 GB card')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard-idx', type=int, default=0,
                    help='for multi-GPU runs: process every --shard-count-th '
                         'source clip starting at this index')
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--raw-dir', default=None,
                    help="override config.yaml's data.raw_dir — use the "
                         "face-masked gesture_training_data_anon here, not "
                         "config.yaml itself, since that path also feeds "
                         "preprocess_holistic.py's real pose extraction, "
                         "which masking can degrade")
    args = ap.parse_args()

    import imageio
    import torch
    import yaml
    from PIL import Image

    from dataset import _extract_frames

    cfg = yaml.safe_load(open(args.config))
    raw = Path(args.raw_dir) if args.raw_dir else Path(cfg['data']['raw_dir'])
    classes = [str(c) for c in cfg['data']['classes']]

    refs = []
    if args.refs:
        refs = sorted(p for p in Path(args.refs).iterdir()
                      if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
        if not refs:
            print(f"no images in {args.refs}")
            return
        print(f"[wan] {len(refs)} reference images")
    elif not args.self_test:
        print("give --refs DIR, or --self-test to validate the pipeline first")
        return

    # `--limit` source clips per class, spread across recording runs so a
    # bulk job does not draw all of them from one sitting. Previously this
    # picked exactly one clip per class regardless of --limit — fine for
    # --self-test, but silent under-generation for a real bulk run: a job
    # asked for --limit 50 would have rendered from 3 source clips, not 50.
    rng = random.Random(args.seed)
    sources = []
    for c in classes:
        vids = sorted((raw / c).glob('*.mp4'))
        if not vids:
            continue
        n = args.limit if args.limit else len(vids)
        by_run = {}
        for v in vids:
            by_run.setdefault(v.stem.split('_')[-2] if '_' in v.stem else '?', []).append(v)
        runs = list(by_run)
        rng.shuffle(runs)
        picks, i = [], 0
        while len(picks) < min(n, len(vids)):
            r = runs[i % len(runs)]
            if by_run[r]:
                picks.append(by_run[r].pop(rng.randrange(len(by_run[r]))))
            i += 1
            if all(not v for v in by_run.values()):
                break
        sources += [(p, c) for p in picks]

    if args.shard_count > 1:
        sources = sources[args.shard_idx::args.shard_count]
        print(f"[shard {args.shard_idx}/{args.shard_count}] {len(sources)} source clips")

    pipe, embeds = load_pipeline(args.steps, bits=args.bits)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    from preprocess_holistic import _init_holistic
    holistic = _init_holistic()

    for src, cls in sources:
        frames = _extract_frames(str(src), num_frames=args.frames)
        if frames is None:
            continue

        H, W = frames.shape[1:3]
        x0, y0, x1, y1 = _compute_crop_box(frames, holistic)
        px0, py0 = int(x0 * W), int(y0 * H)
        px1, py1 = int(x1 * W), int(y1 * H)
        if px1 - px0 >= 16 and py1 - py0 >= 16:
            frames = frames[:, py0:py1, px0:px1, :]

        driving = [Image.fromarray(f) for f in frames]

        if args.self_test:
            ref_list = [("self", Image.fromarray(frames[0]))]
        else:
            chosen = rng.sample(refs, k=min(args.per_clip, len(refs)))
            if args.per_clip > len(refs):
                # more variants requested than distinct reference images —
                # reuse references rather than silently rendering fewer
                chosen += [rng.choice(refs) for _ in range(args.per_clip - len(refs))]
            ref_list = [(r.stem, Image.open(r).convert('RGB')) for r in chosen]

        for ref_name, ref_img in ref_list:
            dest = out / cls
            dest.mkdir(parents=True, exist_ok=True)
            name = f"anim_{src.stem}__{ref_name}.mp4"
            try:
                res = pipe(
                    image=ref_img,
                    driving_video=driving,
                    prompt=PROMPT,
                    negative_prompt=NEGATIVE,
                    **embeds,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    segment_frame_length=args.frames,
                    generator=torch.Generator("cuda").manual_seed(rng.randrange(1 << 30)),
                    output_type="np",
                )
                vid = res.videos[0] if hasattr(res, 'videos') else res.frames[0]
                vid = (np.asarray(vid) * 255).clip(0, 255).astype(np.uint8) \
                    if np.asarray(vid).dtype != np.uint8 else np.asarray(vid)
                imageio.mimsave(str(dest / name), list(vid), fps=args.fps)
                manifest.append({'file': str(dest / name), 'source': str(src),
                                 'label': cls, 'reference': ref_name})
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  wrote {cls:12} {name}  (peak VRAM so far: {peak:.1f} GB)")
            except Exception as e:
                print(f"  FAILED {name}: {type(e).__name__}: {str(e)[:130]}")

    manifest_name = f'manifest_shard{args.shard_idx}.json' if args.shard_count > 1 else 'manifest.json'
    (out / manifest_name).write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} clips -> {out}")
    if manifest:
        print(f"\nvalidate before using any of it:")
        print(f"  python validate_generated.py --dir {out} --by-subdir")


if __name__ == '__main__':
    main()
