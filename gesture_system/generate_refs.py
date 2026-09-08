#!/usr/bin/env python3
"""
generate_refs.py — SDXL reference-person images for animate_variants.py --refs.

Each image gives Wan-Animate a different appearance to swap onto your recorded
motion. Framing matches refs/README.txt: upper body, facing camera, plain
background, arms roughly at rest — the same starting pose your clips have, so
the model isn't asked to reconcile a mismatched pose with the driving video.

Variety across body type, skin tone, clothing, hair, age AND background is the
point: room and lighting is exactly what the video branch was found to key on
instead of the gesture (22.9% balanced accuracy on a held-out session — see
SERVER_SETUP.md), so background variety attacks that shortcut as directly as
appearance variety does.

Backgrounds that include other people keep them blurred/out of focus — a sharp
second face competes with the actual subject for what Wan-Animate treats as
"the person", which would confuse the appearance swap rather than diversify it.

Usage
-----
python generate_refs.py                     # every subject x background pair
python generate_refs.py --per-subject 3     # 3 random backgrounds per subject
python generate_refs.py --limit 5           # quick trial
python generate_refs.py --steps 25 --seed 0
"""

import argparse
import random
import re
from pathlib import Path

FRAMING = (
    "upper body portrait photo, facing camera directly, arms relaxed at sides, "
    "photorealistic, sharp focus on the subject, shot on DSLR"
)
NEGATIVE = (
    "cartoon, illustration, painting, 3d render, deformed, extra limbs, "
    "extra fingers, missing fingers, blurry subject, watermark, text, logo, "
    "cropped head, side profile, looking away, another person in focus"
)

BACKGROUNDS = [
    "plain uncluttered neutral-grey background, even studio lighting",
    "in a modern office, desks and monitors softly out of focus behind them",
    "at a lively party, balloons and blurred guests in the background",
    "in a cozy living room, a couch and bookshelf behind them",
    "in a bright kitchen, cabinets and countertops behind them",
    "outdoors in a park, trees and grass behind them, daylight",
    "in a busy cafe, tables and blurred patrons behind them",
    "in a home gym, exercise equipment behind them",
    "in a classroom, desks and a whiteboard behind them",
    "at an outdoor market, stalls and blurred people behind them",
    "in a garage workshop, tools and shelves behind them",
    "on a rooftop terrace, a city skyline behind them at dusk",
    "in a hotel lobby, modern decor behind them",
    "in a bedroom, a bed and wardrobe behind them",
    "in a conference room, a whiteboard and chairs behind them",
    "in a warehouse, shelving and boxes behind them",
]

# Deliberately varied across body type, skin tone, clothing, hair, age —
# this is the whole point (see module docstring). Add more lines for a
# bigger reference pool; nothing else needs to change.
SUBJECTS = [
    "young Black woman, short curly natural hair, casual blue t-shirt, slim build",
    "middle-aged white man, grey beard, dark green sweater, heavyset build",
    "elderly East Asian woman, silver bob haircut, cream cardigan",
    "young South Asian man, black hair, red hoodie, athletic build",
    "middle-aged Latina woman, long dark hair in a ponytail, mustard yellow blouse",
    "young white man, buzzcut, plain grey t-shirt, tall and lean",
    "middle-aged Black man, shaved head, navy polo shirt, broad build",
    "young East Asian woman, shoulder-length black hair, white blouse",
    "elderly white man, bald, brown flannel shirt, stocky build",
    "young white woman, blonde hair in a bun, olive green sweater",
    "middle-aged South Asian woman, hijab, teal blouse",
    "young Black man, short dreadlocks, mustard yellow hoodie, slim build",
    "middle-aged East Asian man, glasses, short black hair, light blue shirt",
    "young Latina woman, curly brown hair, burgundy top, petite build",
    "elderly Black woman, grey natural hair, purple cardigan",
    "young white man, red hair, plaid flannel shirt, freckles",
    "middle-aged white woman, short grey hair, wine-red sweater",
    "young South Asian woman, long black braid, coral kurta",
    "middle-aged Latino man, moustache, denim jacket over white t-shirt",
    "young white non-binary person, undercut hair dyed teal, black turtleneck",
]


def slug(text, n):
    s = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return f"{n:02d}_{s[:40]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='refs')
    ap.add_argument('--limit', type=int, default=len(SUBJECTS),
                    help='how many subjects to use (not pairs)')
    ap.add_argument('--per-subject', type=int, default=3,
                    help='distinct backgrounds per subject (max 16)')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model', default='stabilityai/stable-diffusion-xl-base-1.0')
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    print(f"[refs] loading {args.model} (fp16)")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to('cuda')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n = 0
    for si, subject in enumerate(SUBJECTS[:args.limit]):
        bgs = rng.sample(BACKGROUNDS, k=min(args.per_subject, len(BACKGROUNDS)))
        for bg in bgs:
            prompt = f"{subject}, {bg}, {FRAMING}"
            gen = torch.Generator('cuda').manual_seed(args.seed + n)
            img = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                       num_inference_steps=args.steps, height=1024, width=768,
                       generator=gen).images[0]
            dest = out / f"{slug(subject, si)}__{slug(bg, n)}.png"
            img.save(dest)
            print(f"  wrote {dest.name}")
            n += 1

    print(f"\n{n} reference images -> {out}/")


if __name__ == '__main__':
    main()
