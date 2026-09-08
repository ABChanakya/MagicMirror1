#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=wan-animate
#SBATCH --time=36:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGE_ME@haw-landshut.de
#
# Runs Wan2.2-Animate-2-14B-Distilled in bf16 — no quantization needed, an
# H100 has 80 GB against the 32.8 GB the transformer needs in bf16. That is
# exactly what blocked this locally: the RTX 4090's 24 GB forced int8
# quantization, which loaded (18.9 GB) but ran out of memory during attention
# at anything above 384x320. Here it runs at the model's native 480x704.
#
# Generation working data (frames, output clips) goes to local /tmp — /scratch
# was the intended local disk but is unreliable for this account on the dgx
# node (see slurm_selftest.sh for why); /tmp is confirmed writable with 1.6 TB
# free. Unlike /scratch, /tmp is not auto-wiped by Slurm, so results are copied
# back to the NFS home and /tmp is removed ourselves before exit.
#
# The 33 GB model itself is cached on NFS home instead (see HF_HOME below),
# not /tmp — a one-time download that persists across jobs, rather than
# repeating on every run. That's a different situation from what README
# section 4 warns against (repeated large I/O against the shared filer during
# a job); this is a single one-time download, then read-only reuse.
#
# Before submitting:
#   1. sed -i 's/CHANGE_ME@haw-landshut.de/your_actual_address/' slurm_animate.sh
#   2. Put reference images (different people, upper body, plain background,
#      facing camera) in gesture_system/refs/ — see refs/README.txt
#   3. sbatch slurm_animate.sh
#
# Check on it:
#   squeue -u $USER
#   tail -f wan-animate_<jobid>.out

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"

SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
echo "OK: using $SCRATCH"

cleanup() {
    echo "== cleaning up $SCRATCH (not auto-wiped like /scratch) =="
    rm -rf "$SCRATCH"
}
trap cleanup EXIT

cd "$GSDIR"
source "$PROJDIR/.venv/bin/activate"

export HF_HOME="$PROJDIR/hf_cache"
mkdir -p "$HF_HOME"

echo "== node/GPU =="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "== generating (2-way sharded across the 2 requested GPUs) =="
# --limit is PER CLASS, not a total. --limit 1000 effectively means "all
# available" (each class has far fewer real clips than that; the script
# caps at what actually exists). Raise LIMIT/PER_CLIP as env vars for a
# bigger run once a pilot pass rate looks right:
#   LIMIT=1000 PER_CLIP=15 sbatch slurm_animate.sh
#
# --fps 30 matches the real clips' actual capture rate (~29 fps) — the
# previous default of 12 would have made generated clips play back in
# noticeable slow motion relative to the real footage they're standing in for.
#
# Each process handles a disjoint 1/NSHARD of the source clips
# (--shard-idx/--shard-count), pinned to its own GPU via
# CUDA_VISIBLE_DEVICES, all writing into the same --out directory — safe
# because filenames are unique per (source clip, reference) pair and each
# shard writes its own manifest_shard<i>.json instead of a shared manifest.json.
NSHARD=2
pids=()
for i in $(seq 0 $((NSHARD - 1))); do
    CUDA_VISIBLE_DEVICES=$i python animate_variants.py \
        --refs refs/ \
        --raw-dir ../camera/gesture_training_data_anon \
        --out "$SCRATCH/animated" \
        --bits 0 \
        --height 240 --width 320 \
        --frames 30 --steps 10 --fps 30 \
        --limit "${LIMIT:-1000}" \
        --per-clip "${PER_CLIP:-1}" \
        --shard-idx "$i" --shard-count "$NSHARD" \
        > "$GSDIR/shard${i}_${SLURM_JOB_ID}.out" 2>&1 &
    pids+=($!)
    echo "  launched shard $i on GPU $i (pid $!)"
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" = "1" ]; then
    echo "WARNING: at least one shard exited non-zero — check shard*_${SLURM_JOB_ID}.out"
fi

echo "== copying results back to NFS home =="
# Before validating, not after: /tmp is wiped by our own cleanup trap on exit,
# so a validate_generated.py crash used to take all the generated clips down
# with it (a mediapipe packaging issue destroyed a full batch this way).
mkdir -p "$GSDIR/animated_results"
cp -r "$SCRATCH/animated" "$GSDIR/animated_results/job_${SLURM_JOB_ID}"

echo "== validating (on the saved copy, not \$SCRATCH) =="
python validate_generated.py --dir "$GSDIR/animated_results/job_${SLURM_JOB_ID}" --by-subdir || true

echo "Job done. Results in $GSDIR/animated_results/job_${SLURM_JOB_ID}"
echo "Re-run validation any time with:"
echo "  python validate_generated.py --dir $GSDIR/animated_results/job_${SLURM_JOB_ID} --by-subdir --move-rejects"
