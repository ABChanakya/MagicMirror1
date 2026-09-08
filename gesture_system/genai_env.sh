# Source before generating:  source genai_env.sh
#
# Model weights and generated clips go to the 7.3 TB HDD, not to / (3.8 GB free).
# The HDD is NTFS but ntfs-3g supports symlinks, so the HuggingFace cache layout
# works unmodified. If it is ever remounted without symlink support, set
# HF_HUB_DISABLE_SYMLINKS=1 and the cache falls back to copying.
export HF_HOME=/media/bhaskara/Volume/hf_cache
export HUGGINGFACE_HUB_CACHE=/media/bhaskara/Volume/hf_cache/hub
export GEN_OUT=/media/bhaskara/Volume/generated
echo "HF_HOME=$HF_HOME"
df -h /media/bhaskara/Volume | tail -1
