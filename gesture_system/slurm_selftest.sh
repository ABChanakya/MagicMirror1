#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=animate-selftest
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# One-clip smoke test, run as a batch job rather than interactively.
#
# /scratch's per-job directory is unreliable for this account on the dgx
# node: job 3203 saw mkdir succeed, then the exact same path fail with
# Permission denied moments later in the same run — something (likely a
# Prolog process that also can't resolve this account's username; `id` shows
# the group name resolves but the user name doesn't) is racing against and
# reclaiming it mid-job. Rather than chase that, this always uses
# /tmp/$SLURM_JOB_ID instead, confirmed writable with 1.6 TB free. Unlike
# /scratch, /tmp is NOT auto-wiped by Slurm on job end, so the trap below
# removes it ourselves.
#
# Submit with:  sbatch slurm_selftest.sh
# Watch with:   squeue -u $USER
#               tail -f animate-selftest_<jobid>.out

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"

echo "== identity check (whoami/getent may fail — harmless, see comment above) =="
whoami || true
id || true

echo "== node/GPU =="
hostname
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "== local scratch (/tmp, not /scratch) =="
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

# On NFS home, not $SCRATCH: the 33 GB model download would otherwise repeat
# on every single job, since /tmp gets wiped by our own cleanup trap above.
# One persistent copy here means later jobs skip straight to generation.
export HF_HOME="$PROJDIR/hf_cache"
mkdir -p "$HF_HOME"

echo "== self-test: one clip, reference = its own first frame =="
# If this reproduces the original gesture, the pipeline (model load, bf16,
# driving-video conditioning) is sound and the only remaining variable for
# the real run is the reference images.
python animate_variants.py --self-test --limit 1 --bits 0 \
    --raw-dir ../camera/gesture_training_data_anon \
    --height 240 --width 320 --frames 30 --steps 10 \
    --out "$SCRATCH/selftest_out"

echo "== copying result back to NFS home =="
# Before validating, not after: /tmp is wiped by our cleanup trap on exit, so
# a validate_generated.py crash would take the generated clip down with it.
mkdir -p "$GSDIR/selftest_results"
cp -r "$SCRATCH/selftest_out" "$GSDIR/selftest_results/job_${SLURM_JOB_ID}"

echo "== validating (on the saved copy, not \$SCRATCH) =="
python validate_generated.py --dir "$GSDIR/selftest_results/job_${SLURM_JOB_ID}" --by-subdir || true

echo "Done. Check $GSDIR/selftest_results/job_${SLURM_JOB_ID}/"
