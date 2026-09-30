#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=kimodo-constrained-test
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Test batch for keyframe-constrained KiMoDo generation: for each real-clip
# constraint target (3 swipe_right, 1 swipe_left, x2 for --flip-x/noflip),
# solve IK for constraints.json, then generate. Runs on an actual GPU
# compute node (unlike the earlier interactive attempt on the login node,
# which got OOM-killed loading the 8B-param text encoder there).
#
# Submit with: sbatch slurm_kimodo_constrained_test.sh

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"
KVENV="$PROJDIR/kimodo_venv"

cd "$GSDIR"
source "$KVENV/bin/activate"
export HF_HOME="$PROJDIR/hf_cache"

PROMPT_RIGHT="A person stands facing forward and swipes one hand to the right in a single motion, as if wiping something off a table or pushing it aside."
PROMPT_LEFT="A person stands facing forward and swipes one hand to the left in a single motion, as if wiping something off a table or pushing it aside."

for target in constraint_target_swipe_*.npz constraint_v2_*.npz; do
    [ -f "$target" ] || continue
    case "$target" in
        *_gen.npz) continue ;;  # skip generated outputs, only process input targets
    esac
    base=$(basename "$target" .npz)
    out="${base}_gen"
    [ -f "${out}.npz" ] && { echo "skip $base (already generated)"; continue; }

    echo "== solving IK: $base =="
    python3 kimodo_solve_constraint.py --target "$target" --out "${base}_constraints.json"

    case "$base" in
        *swipe_right*) prompt="$PROMPT_RIGHT" ;;
        *swipe_left*) prompt="$PROMPT_LEFT" ;;
    esac

    echo "== generating: $base =="
    kimodo_gen "$prompt" --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed 1 \
        --constraints "${base}_constraints.json" --output "$out"
done

echo "== output files =="
ls -la constraint_target_swipe_*_gen.npz constraint_v2_*_gen.npz 2>/dev/null

echo "Done."
