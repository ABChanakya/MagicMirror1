#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=kimodo-prompttest
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Small (15/direction) test of a tweaked prompt before committing the full
# 200/class run in slurm_kimodo_generate.sh. Original prompt's "wiping
# something off a table" framing produces some clips that reach up and loop
# rather than sweep flat (e.g. kimodo_swipe_right_1.npz: right wrist travels
# 0.72 in Y vs 0.56 in X -- more vertical than horizontal, axis_ratio 0.1).
# This drops the table framing and adds an explicit constant-height,
# flat-motion constraint to see if it reduces that failure mode.
#
# Writes to kimodo_test2_swipe_{left,right}_N.npz -- separate namespace
# from the main seeds 1-200, so it can't collide with or get counted
# alongside the committed batch.
#
# Submit with: sbatch slurm_kimodo_prompt_test.sh

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"
KVENV="$PROJDIR/kimodo_venv"

cd "$GSDIR"
source "$KVENV/bin/activate"

export CPATH="$HOME/local_pydev/extracted/usr/include/python3.12:$HOME/local_pydev/extracted/usr/include${CPATH:+:$CPATH}"
export CMAKE_PREFIX_PATH="$HOME/local_pydev/extracted/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_INCLUDE_PATH="$HOME/local_pydev/extracted/usr/include/python3.12:$HOME/local_pydev/extracted/usr/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$HOME/local_pydev/extracted/usr/lib/x86_64-linux-gnu${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
export HF_HOME="$PROJDIR/hf_cache"

echo "== generating: swipe left test prompt (x15) =="
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    kimodo_gen "A person stands facing forward with the hand at chest height and sweeps it sideways to the left in a single flat, horizontal motion, without raising or lowering the hand." \
        --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed "$i" \
        --output "$GSDIR/kimodo_test2_swipe_left_${i}"
done

echo "== generating: swipe right test prompt (x15) =="
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    kimodo_gen "A person stands facing forward with the hand at chest height and sweeps it sideways to the right in a single flat, horizontal motion, without raising or lowering the hand." \
        --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed "$i" \
        --output "$GSDIR/kimodo_test2_swipe_right_${i}"
done

echo "== output files =="
ls -la "$GSDIR"/kimodo_test2_swipe_*.npz

echo "Done."
