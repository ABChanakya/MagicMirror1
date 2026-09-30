#!/usr/bin/env python3
"""
visualize_kimodo.py — render a KiMoDo .npz motion as a stick-figure GIF, no
diffusion/rendering model involved. Shows all 3 orthographic projections
(XY, ZY, XZ) side by side since we don't know a priori which two axes are
KiMoDo's "screen" plane and which is depth — whichever projection looks
like a standing person facing the camera is the one that matters, and its
horizontal axis is the one kimodo_to_landmarks.py's dx is reading.

Each wrist leaves a solid trail of its full path so the swipe direction is
visible even from a single still frame, not just by watching it play.

Usage
-----
python visualize_kimodo.py --npz kimodo_swipe_right_6.npz --out swipe_right_6.gif
"""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from kimodo_to_landmarks import SOMA

BONES = [
    (SOMA['l_shoulder'], SOMA['l_elbow']), (SOMA['l_elbow'], SOMA['l_wrist']),
    (SOMA['r_shoulder'], SOMA['r_elbow']), (SOMA['r_elbow'], SOMA['r_wrist']),
    (SOMA['l_shoulder'], SOMA['r_shoulder']),
    (SOMA['l_hip'], SOMA['r_hip']),
    (SOMA['l_shoulder'], SOMA['l_hip']), (SOMA['r_shoulder'], SOMA['r_hip']),
]
VIEWS = [('Front: X-Y', 0, 1), ('Side: Z-Y', 2, 1), ('Top: X-Z', 0, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True, help='output .gif path')
    ap.add_argument('--fps', type=int, default=8)
    ap.add_argument('--title', default=None)
    args = ap.parse_args()

    d = np.load(args.npz)
    posed = d['posed_joints']  # (T, 77, 3)
    T = posed.shape[0]
    l_wrist_path = posed[:, SOMA['l_wrist']]
    r_wrist_path = posed[:, SOMA['r_wrist']]

    lo = posed.min(axis=(0, 1)) - 0.15
    hi = posed.max(axis=(0, 1)) + 0.15

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(args.title or args.npz)

    artists = {}
    for ax, (title, xi, yi) in zip(axes, VIEWS):
        ax.set_title(title)
        ax.set_xlim(lo[xi], hi[xi])
        ax.set_ylim(lo[yi], hi[yi])  # KiMoDo is Y-up (shoulders > hips in raw Y), no flip needed
        ax.set_aspect('equal')
        ax.grid(alpha=0.2)
        body_sc = ax.scatter([], [], s=10, c='tab:blue', zorder=3)
        l_wrist_sc = ax.scatter([], [], s=60, c='tab:green', zorder=5, label='L wrist')
        r_wrist_sc = ax.scatter([], [], s=60, c='tab:orange', zorder=5, label='R wrist')
        bone_lines = [ax.plot([], [], '-', c='gray', lw=2, zorder=2)[0] for _ in BONES]
        l_trail, = ax.plot([], [], '-', c='tab:green', lw=1.5, alpha=0.7, zorder=1)
        r_trail, = ax.plot([], [], '-', c='tab:orange', lw=1.5, alpha=0.7, zorder=1)
        ax.legend(loc='upper right', fontsize=8)
        artists[(xi, yi)] = (body_sc, l_wrist_sc, r_wrist_sc, bone_lines, l_trail, r_trail)

    frame_text = fig.text(0.5, 0.02, '', ha='center')

    def update(t):
        frame = posed[t]
        changed = []
        for (xi, yi), (body_sc, l_wrist_sc, r_wrist_sc, bone_lines, l_trail, r_trail) in artists.items():
            body_sc.set_offsets(frame[:, [xi, yi]])
            l_wrist_sc.set_offsets(frame[SOMA['l_wrist'], [xi, yi]][None, :])
            r_wrist_sc.set_offsets(frame[SOMA['r_wrist'], [xi, yi]][None, :])
            for (a, b), ln in zip(BONES, bone_lines):
                ln.set_data([frame[a, xi], frame[b, xi]], [frame[a, yi], frame[b, yi]])
            l_trail.set_data(l_wrist_path[:t + 1, xi], l_wrist_path[:t + 1, yi])
            r_trail.set_data(r_wrist_path[:t + 1, xi], r_wrist_path[:t + 1, yi])
            changed += [body_sc, l_wrist_sc, r_wrist_sc, l_trail, r_trail] + bone_lines
        frame_text.set_text(f'frame {t + 1}/{T}')
        return changed

    ani = animation.FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(args.out, writer='pillow', fps=args.fps)
    plt.close(fig)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
