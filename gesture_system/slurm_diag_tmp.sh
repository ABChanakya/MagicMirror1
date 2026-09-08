#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=diag-tmp
#SBATCH --time=00:05:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# One-off diagnostic: does this account get a working /scratch on the dgx
# node, and is /tmp a viable fallback if not? Deliberately has NO `set -e`
# and every step is guarded, so it runs to completion and reports on
# everything instead of dying at the first failure (which is exactly what
# hid the actual cause of the /scratch failures in earlier jobs).
#
# Submit with: sbatch slurm_diag_tmp.sh
# Then:        cat diag-tmp_<jobid>.out

echo "== identity =="
echo "id -u:  $(id -u 2>&1)"
echo "id -g:  $(id -g 2>&1)"
whoami 2>&1 || echo "(whoami failed, as expected)"
getent passwd "$(id -u)" 2>&1 || echo "(getent by uid failed)"
getent passwd "$USER" 2>&1 || echo "(getent by name failed)"

echo
echo "== node =="
hostname

echo
echo "== /scratch =="
ls -ld /scratch 2>&1
SCRATCH="/scratch/${USER}_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH" 2>&1 && echo "OK: created $SCRATCH" || echo "FAILED: could not create $SCRATCH"

echo
echo "== /tmp fallback =="
ls -ld /tmp 2>&1
df -h /tmp 2>&1
TMPSCRATCH="/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPSCRATCH" 2>&1 && echo "OK: created $TMPSCRATCH" || echo "FAILED: could not create $TMPSCRATCH"
if [ -d "$TMPSCRATCH" ]; then
    dd if=/dev/zero of="$TMPSCRATCH/writetest" bs=1M count=100 2>&1 && echo "OK: wrote 100MB test file" || echo "FAILED: write test"
    rm -rf "$TMPSCRATCH"
fi

echo
echo "== \$HOME (NFS) sanity check =="
df -h "$HOME" 2>&1

echo
echo "== done =="
