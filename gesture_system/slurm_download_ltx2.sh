#!/bin/bash
#SBATCH --partition=dgx_01
#SBATCH --qos=research_qos
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=ltx2-download
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Pure download job — no GPU compute, just network + disk. Requests a
# minimal GPU anyway since dgx_01 jobs need a gres allocation; the point is
# to get the ~145 GB LTX-2 diffusers folder + the pose-control LoRA onto
# NFS home (persistent HF_HOME, same cache Wan-Animate uses) BEFORE we
# spend GPU-hours on an actual generation job. Loading a 100 GB text
# encoder needs enable_sequential_cpu_offload() at inference time — that's
# a diffusers-side concern, not a download-time one.
#
# Submit with: sbatch slurm_download_ltx2.sh
# Watch with:  squeue -u $USER
#              tail -f ltx2-download_<jobid>.out

set -euo pipefail

PROJDIR="$HOME/magicmirror"
GSDIR="$PROJDIR/gesture_system"

cd "$GSDIR"
source "$PROJDIR/.venv/bin/activate"

export HF_HOME="$PROJDIR/hf_cache"
mkdir -p "$HF_HOME"

echo "== downloading Lightricks/LTX-2 (diffusers folder, ~145 GB) =="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Lightricks/LTX-2',
    allow_patterns=[
        'transformer/*', 'text_encoder/*', 'vae/*', 'connectors/*',
        'latent_upsampler/*', 'vocoder/*', 'audio_vae/*', 'tokenizer/*',
        'scheduler/*', 'model_index.json',
    ],
)
"

echo "== downloading pose-control IC-LoRA (small) =="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Lightricks/LTX-2-19b-IC-LoRA-Pose-Control')
"

echo "Done. Cached under \$HF_HOME = $HF_HOME"
du -sh "$HF_HOME"
