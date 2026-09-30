#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=kimodo-expmatrix
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# 3x3 experiment: keyframe count (none / 2pt / multi=4) x prompt style
# (generic / hand-named / direction-neutral), 8 real source clips each =
# 72 generations. Natural hand only (not re-testing cross-hand here).
#
# Keyframe conditions:
#   none  -- plain text-to-motion, no --constraints at all (today's
#            original approach, the baseline every other test compares to)
#   2pt   -- constrained to just the real clip's start and end position
#   multi -- constrained to start + 2 middle + end (4 keyframes)
#
# Prompt conditions:
#   generic -- today's wording, e.g. "swipes one hand to the right"
#   named   -- explicitly states which hand, e.g. "swipes their right hand
#              to the right" (tests whether naming the hand -- even the
#              already-correct natural hand -- changes anything)
#   neutral -- identical wording regardless of direction, e.g. "sweeps one
#              hand across their body" -- for constrained modes this tests
#              whether the constraint alone can carry direction without the
#              text's help; for 'none' mode this is a deliberate null case,
#              since nothing at all would tell the model which way to go
#
# Requires exp_targets/exp_{label}_{idx}_{2pt,multi}.npz already built
# locally via build_constraint_targets.py and synced over.
#
# Submit with: sbatch slurm_kimodo_experiment_matrix.sh

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"
KVENV="$PROJDIR/kimodo_venv"

cd "$GSDIR"
source "$KVENV/bin/activate"
export HF_HOME="$PROJDIR/hf_cache"

CLIPS="swipe_right:471 swipe_right:472 swipe_right:473 swipe_right:340 swipe_left:171 swipe_left:79 swipe_left:99 swipe_left:186"

for entry in $CLIPS; do
    label="${entry%%:*}"
    idx="${entry##*:}"
    natural_hand=$([ "$label" = "swipe_right" ] && echo right || echo left)
    direction=$([ "$label" = "swipe_right" ] && echo right || echo left)

    prompt_generic="A person stands facing forward and swipes one hand to the ${direction} in a single motion, as if wiping something off a table or pushing it aside."
    prompt_named="A person stands facing forward and swipes their ${natural_hand} hand to the ${direction} in a single motion, as if wiping something off a table or pushing it aside."
    prompt_neutral="A person stands facing forward and sweeps one hand across their body in a single motion, as if wiping something off a table or pushing something aside."

    for kf_mode in none 2pt multi; do
        for p_mode in generic named neutral; do
            base="exp_${label}_${idx}_${kf_mode}_${p_mode}"
            out="${base}_gen"
            [ -f "${out}.npz" ] && { echo "skip $base (already generated)"; continue; }

            case "$p_mode" in
                generic) prompt="$prompt_generic" ;;
                named)   prompt="$prompt_named" ;;
                neutral) prompt="$prompt_neutral" ;;
            esac

            if [ "$kf_mode" = "none" ]; then
                echo "== generating: $base (no constraint) =="
                kimodo_gen "$prompt" --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed 1 --output "$out"
            else
                target="exp_targets/exp_${label}_${idx}_${kf_mode}.npz"
                echo "== solving IK: $base =="
                python3 kimodo_solve_constraint.py --target "$target" --out "${base}_constraints.json"
                echo "== generating: $base =="
                kimodo_gen "$prompt" --model Kimodo-SOMA-RP-v1 --duration 2.0 --seed 1 \
                    --constraints "${base}_constraints.json" --output "$out"
            fi
        done
    done
done

echo "== output files =="
ls exp_*_gen.npz 2>/dev/null | wc -l

echo "Done."
