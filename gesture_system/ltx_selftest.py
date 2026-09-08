#!/usr/bin/env python3
"""
ltx_selftest.py — one-clip smoke test for LTX-2 IC-LoRA Pose Control.

Same self-test contract as animate_variants.py --self-test: reference image =
the clip's own first frame, so if the output reproduces the original gesture,
the pipeline (skeleton rendering, LoRA loading, motion conditioning) is sound
and the only remaining variable for a real run is the reference images.

Reuses render_pose_video.py's raw_landmarks()/render() directly — the pose
video it draws is already face-free by construction (Holistic extraction in
preprocess_holistic.py never requests face landmarks), so no separate
face-data handling is needed here at all, unlike WanAnimatePipeline's
face_video requirement.

Usage
-----
python ltx_selftest.py --clip swipe_left/swipe_left_chanakya_20260816-165535_103
"""

import argparse

import cv2
import numpy as np
import yaml
from PIL import Image

MID = "Lightricks/LTX-2"
LORA_ID = "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control"
LORA_FILE = "ltx-2-19b-ic-lora-pose-control.safetensors"

PROMPT = (
    "A person from the waist up, facing the camera, plain background, "
    "raising one hand and sweeping it smoothly across the frame in a single "
    "deliberate motion, natural human proportions, five fingers per hand, "
    "sharp focus, static locked-off camera, realistic photographic lighting"
)
NEGATIVE = (
    "shaky, glitchy, low quality, worst quality, deformed, distorted, "
    "disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, "
    "weird hand, extra limbs, extra fingers, missing fingers, transition, static"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--clip', required=True, help='<class>/<stem>, e.g. swipe_left/foo')
    ap.add_argument('--raw-dir', default='../camera/gesture_training_data_anon')
    ap.add_argument('--out', default='ltx_selftest_out.mp4')
    ap.add_argument('--frames', type=int, default=25,
                    help='LTX wants num_frames = 8n+1; 25 = 8*3+1')
    ap.add_argument('--height', type=int, default=320)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance-scale', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import torch
    from diffusers import LTX2InContextPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    from render_pose_video import raw_landmarks, render
    from pathlib import Path

    cfg = yaml.safe_load(open(args.config))
    cls, stem = args.clip.split('/', 1)
    src = Path(args.raw_dir) / cls / f"{stem}.mp4"
    print(f"[ltx] source clip: {src}")

    print("[ltx] extracting + rendering pose skeleton (face-free, no separate face data)")
    xy = raw_landmarks(src, args.frames)
    if xy is None:
        raise SystemExit(f"could not read {src}")
    detected = np.isfinite(xy[:, 4, 0]).mean()  # left-shoulder index within kept pose block
    print(f"  pose detected in {detected*100:.0f}% of frames")
    skeleton_bgr = render(xy, args.width, args.height)
    skeleton_rgb = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in skeleton_bgr]

    # self-test: reference image = the clip's own first real frame
    from dataset import _extract_frames
    real_frames = _extract_frames(str(src), num_frames=args.frames)
    ref_img = Image.fromarray(real_frames[0]).resize((args.width, args.height))

    print(f"[ltx] loading {MID} (bf16, sequential CPU offload — text encoder alone is ~100 GB)")
    pipe = LTX2InContextPipeline.from_pretrained(MID, torch_dtype=torch.bfloat16)
    print(f"[ltx] loading IC-LoRA pose control adapter")
    pipe.load_lora_weights(LORA_ID, weight_name=LORA_FILE)
    pipe.enable_sequential_cpu_offload(device="cuda:0")

    gen = torch.Generator("cuda").manual_seed(args.seed)
    print("[ltx] generating (sequential offload is slow — expect several minutes)")
    # Single-stage call, not the README's full two-stage (latent -> upsample ->
    # distilled refine) production pipeline — that's for final visual quality;
    # this smoke test only needs to know whether pose conditioning follows the
    # skeleton's motion at all before investing in the heavier setup.
    video, audio = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        reference_conditions=[LTX2ReferenceCondition(frames=skeleton_rgb)],
        conditions=[LTX2VideoCondition(frames=[ref_img], index=0)],
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
        output_type="np",
        return_dict=False,
    )
    video = (np.asarray(video[0]) * 255).clip(0, 255).astype(np.uint8)

    import imageio
    imageio.mimsave(args.out, list(video), fps=12)
    print(f"[ltx] wrote {args.out}")


if __name__ == '__main__':
    main()
