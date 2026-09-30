#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=kimodo-gen
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Generates KiMoDo 3D motion .npz files directly — no rendering, no video
# model. Feeds kimodo_to_landmarks.py, which maps these onto the project's
# own 174-dim landmark features and validates the wrist trajectory in
# skeleton-space (see that script's docstring for why no camera projection
# is needed).
#
# Reuses fixes already worked out for this exact install:
#   - missing python3.12-dev headers (no root) -> locally extracted .deb,
#     pointed at via CMAKE_PREFIX_PATH/CMAKE_INCLUDE_PATH/CMAKE_LIBRARY_PATH
#   - peft required for load_lora_weights-style adapters kimodo itself uses
#   - meta-llama/Meta-Llama-3-8B-Instruct (gated, for the local LLM2Vec
#     text-encoder fallback) -- token already saved at
#     $PROJDIR/hf_cache/token from the earlier `hf auth login`, so this
#     should just work without repeating that whole setup.
#
# Submit with: sbatch slurm_kimodo_generate.sh

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"
KVENV="$PROJDIR/kimodo_venv"

cd "$GSDIR"

echo "== node/GPU =="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

if [ ! -d "$KVENV" ]; then
    echo "== creating kimodo venv =="
    python3 -m venv "$KVENV"
    source "$KVENV/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    source "$KVENV/bin/activate"
fi

export CPATH="$HOME/local_pydev/extracted/usr/include/python3.12:$HOME/local_pydev/extracted/usr/include${CPATH:+:$CPATH}"
export CMAKE_PREFIX_PATH="$HOME/local_pydev/extracted/usr${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_INCLUDE_PATH="$HOME/local_pydev/extracted/usr/include/python3.12:$HOME/local_pydev/extracted/usr/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}"
export CMAKE_LIBRARY_PATH="$HOME/local_pydev/extracted/usr/lib/x86_64-linux-gnu${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
pip install "kimodo[all] @ git+https://github.com/nv-tlabs/kimodo.git"

export HF_HOME="$PROJDIR/hf_cache"

echo "== generating: swipe left (x200) =="
for i in $(seq 1 200); do
    out="$GSDIR/kimodo_swipe_left_${i}"
    [ -f "${out}.npz" ] && continue
    kimodo_gen "A person stands facing forward and swipes one hand to the left in a single motion, as if wiping something off a table or pushing it aside." \
        --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed "$i" \
        --output "$out"
done

echo "== generating: swipe right (x200) =="
for i in $(seq 1 200); do
    out="$GSDIR/kimodo_swipe_right_${i}"
    [ -f "${out}.npz" ] && continue
    kimodo_gen "A person stands facing forward and swipes one hand to the right in a single motion, as if wiping something off a table or pushing it aside." \
        --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed "$i" \
        --output "$out"
done

echo "== generating: null / idle motion (x100) =="
for i in $(seq 1 100); do
    out="$GSDIR/kimodo_null_${i}"
    [ -f "${out}.npz" ] && continue
    kimodo_gen "A person stands facing forward, casually shifting their weight and adjusting their stance, with no hand gestures and no swiping or pointing motions." \
        --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed "$i" \
        --output "$out"
done

echo "== output files =="
ls -la "$GSDIR"/kimodo_swipe_*.npz "$GSDIR"/kimodo_null_*.npz

echo "Done."
