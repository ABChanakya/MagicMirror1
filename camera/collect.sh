#!/usr/bin/env bash
# collect.sh — launch the gesture collector with the repo venv
#
#   ./collect.sh cameras                # pick which camera to use, by eye
#   ./collect.sh                        # interactive menu
#   ./collect.sh batch swipe_left 25    # straight into a batch
#   ./collect.sh status
#
# Run `cameras` once (and again after replugging the webcam) — the choice is
# remembered in .camera_choice.json.
#
# Set GESTURE_SUBJECT so clips are tagged with who performed them:
#   GESTURE_SUBJECT=chanakya ./collect.sh
set -euo pipefail

cd "$(dirname "$0")"

VENV="../.venv/bin/python"
if [ ! -x "$VENV" ]; then
    if [ -x ".venv/bin/python" ]; then
        VENV=".venv/bin/python"
    else
        echo "No venv found. Create one at the repo root:"
        echo "  python3 -m venv .venv && .venv/bin/pip install -r camera/requirements.txt"
        exit 1
    fi
fi

export GESTURE_SUBJECT="${GESTURE_SUBJECT:-unknown}"

# Silence MediaPipe/TF/absl chatter at the source. Filtering stderr through a
# pipe instead would run it asynchronously, which scrambles the ordering of the
# interactive prompts.
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=2
export GLOG_logtostderr=0
export ABSL_LOGGING_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning"

exec "$VENV" collect_gesture_data.py "$@"
