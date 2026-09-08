#!/usr/bin/env bash
#
# cleanup.sh — remove junk and duplicates under Magicmirror3/
#
# DRY RUN BY DEFAULT.  ./cleanup.sh --apply   actually deletes.
#
# Two sections:
#   TIER 1  safe to delete   — regenerable or verified byte-identical
#   TIER 3  superseded       — has a named successor or a concluded experiment
#   TIER 2  needs review     — printed only, NEVER deleted by this script
#
# Never touched: .git/ .venv/ node_modules/ site-packages/ archive/
#                camera/gesture_training_data/   (932 recorded clips, irreplaceable)
#                gesture_system/data/landmarks_holistic.npz  (current cache)
#
# Scan basis: 3427 user files hashed (sha256), recursive.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

# Tier 2 is print-only and is never deleted by this script, whatever the flags.
APPLY=0; DO1=1; DO3=1
case "${1:-}" in
  --apply)        APPLY=1 ;;                 # tiers 1 and 3
  --apply-tier1)  APPLY=1; DO3=0 ;;
  --apply-tier3)  APPLY=1; DO1=0 ;;
  --tier1)        DO3=0 ;;                   # dry run, tier 1 only
  --tier3)        DO1=0 ;;
  "" ) ;;
  *) echo "usage: $0 [--apply | --apply-tier1 | --apply-tier3 | --tier1 | --tier3]"; exit 1 ;;
esac
if (( APPLY )); then
  t=""; (( DO1 )) && t="1"; (( DO3 )) && t="${t:+$t and }3"
  echo "APPLYING — deleting tier $t"
else
  echo "DRY RUN — nothing deleted."
fi
echo

freed=0
note() { printf '  %-52s %8s  %s\n' "$1" "$2" "${3:-}"; }

rm_path() {  # path, reason
  local p="$1" why="$2" sz
  [[ -e "$p" ]] || return 0
  sz=$(du -sb "$p" 2>/dev/null | cut -f1); sz=${sz:-0}
  note "$p" "$(numfmt --to=iec "$sz")" "$why"
  (( APPLY )) && rm -rf -- "$p"
  freed=$((freed + sz))
}

rm_dupe() {  # keep, drop, reason — re-verifies hashes before acting
  local keep="$1" drop="$2" why="$3"
  [[ -f "$keep" && -f "$drop" ]] || return 0
  local hk hd sz
  hk=$(sha256sum "$keep" | cut -d' ' -f1)
  hd=$(sha256sum "$drop" | cut -d' ' -f1)
  if [[ "$hk" != "$hd" ]]; then
    note "$drop" "-" "SKIPPED: no longer identical to $keep"
    return 0
  fi
  sz=$(stat -c%s "$drop")
  note "$drop" "$(numfmt --to=iec "$sz")" "$why (keeping $keep)"
  (( APPLY )) && rm -f -- "$drop"
  freed=$((freed + sz))
}

# ===========================================================================
if (( DO1 )); then
echo "TIER 1 — SAFE TO DELETE"
echo "───────────────────────────────────────────────────────────────────────"
# __pycache__ is deliberately left alone (kept at your request). It is only
# ~250 KB and removing it costs a slower first import for no real gain.
# To include it, uncomment:
#
# while IFS= read -r d; do rm_path "$d" "python bytecode cache"; done < <(
#   find . -type d -name __pycache__ \
#        -not -path './.git/*' -not -path '*/node_modules/*' \
#        -not -path '*/.venv/*' -not -path '*/site-packages/*' 2>/dev/null)

# ---------------------------------------------------------------------------
# Run logs from this project's own scripts. All are .gitignored already, and
# every one is reproducible by re-running the script that wrote it.
# Kept deliberately: scaling_results*.json, twostream_results.json,
# experiment_results.json, review_verdicts.json — those are RESULTS, not logs.
# ---------------------------------------------------------------------------
for f in gesture_system/*.log logs/*.log MagicMirror/logs/*.log; do
  [[ -e "$f" ]] && rm_path "$f" "regenerable run log"
done

# ---------------------------------------------------------------------------
# Verified byte-identical duplicates (hash re-checked at delete time).
# ---------------------------------------------------------------------------
rm_dupe "camera/yolov8n-pose.pt" "MagicMirror/yolov8n-pose.pt" \
        "duplicate YOLO weights; no code references either path"
rm_dupe "typescript" "MagicMirror/typescript" \
        "stray 'script' terminal capture committed twice"
fi   # end tier 1

echo
# ===========================================================================
if (( DO3 )); then
echo "TIER 3 — SUPERSEDED SCRIPTS & OUTPUTS"
echo "───────────────────────────────────────────────────────────────────────"
# Each of these has a named successor or a concluded experiment behind it.
# Nothing imports them and no shell script or doc invokes them.

# --- gesture_system: replaced by a newer file ---
rm_path gesture_system/preprocess_landmarks.py "-> preprocess_holistic.py (66-dim -> 174-dim)"
rm_path gesture_system/preaugment.py           "-> on-the-fly augmentation; this caused the val/train leak"
rm_path gesture_system/sweep.py                "-> experiments.py (staged CV search)"
rm_path gesture_system/hparam_search.py        "old 5-class Optuna search"
rm_path gesture_system/train.py                "old 5-class two-stream trainer"
rm_path gesture_system/verify_inference.py     "one-off check, superseded by test_pipeline.py"

# --- gesture_system: experiments that reached a conclusion ---
rm_path gesture_system/add_velocity.py         "velocity channels measured: no effect (64.2 vs 64.7, inside noise)"
rm_path gesture_system/check_tracking.py       "answered: motion is not why MediaPipe drops frames"
rm_path gesture_system/check_weighting.py      "written but never run"
rm_path gesture_system/run_overnight.sh        "old batch runner; superseded by direct invocation"

# --- camera: replaced by collect_gesture_data.py ---
rm_path camera/record_auto.py                  "-> collect_gesture_data.py"
rm_path camera/record_batch.py                 "-> collect_gesture_data.py"
rm_path camera/train.py                        "old face-recognition trainer"
rm_path camera/debug_gesture_live.py           "debug tool for the pre-rewrite pipeline"
rm_path camera/merge_incoming.py               "cross-machine merge; recording is local now"

# --- stray artefacts ---
rm_path typescript                             "captured terminal session from 2026-03-27"
rm_path camera/training_history.png            "plot from the removed camera/train.py"
rm_path camera/training_history_motion.png     "plot from the removed camera/train.py"
rm_path gesture_system/scaling_results.json    "superseded by scaling_results_v2.json"

# --- superseded landmark caches (regenerable; preprocess_holistic.py, ~20 min) ---
rm_path gesture_system/data/landmarks.npz              "66-dim hand-only encoding, superseded"
rm_path gesture_system/data/landmarks_holistic_vel.npz "velocity variant; experiment concluded no effect"
fi   # end tier 3

echo
# ===========================================================================
echo "TIER 2 — NEEDS REVIEW  (not deleted; decide per item)"
echo "───────────────────────────────────────────────────────────────────────"

cat <<'REVIEW'
  gesture_system/checkpoints/best_model.pt                    384 MB
      5-class Video Swin model from the ABANDONED two-stream approach.
      Superseded: current pipeline is 3-class landmark-only. Almost certainly
      deletable, but it is the only artefact of that experiment.

  gesture_system/checkpoints/twostream_video.pt               365 MB
  gesture_system/checkpoints/twostream_fusion.pt              370 MB
      Video-only (25.4% balanced) and fusion (26.3%) — both BELOW the 33%
      chance line. Kept only as evidence for the shortcut-learning finding.
      If that result is written up, the numbers suffice and these can go.
      -> deleting all three above frees ~1.1 GB of the 1.2 GB checkpoints dir.

  gesture_system/checkpoints/landmark_best.pt                 5.6 MB
      KEEP. Current best model, 86.8% balanced, 3-class. This is the one
      inference.py loads.

  gesture_system/checkpoints/{novel,vel}/twostream_landmark.pt  5.8 MB each
      From the velocity-channel A/B. That experiment concluded "no effect"
      (64.2% vs 64.7%, inside noise), so these are spent — but they are the
      raw evidence for it.

  gesture_system/checkpoints/landmark_noweight.pt             2.6 MB
      Class-weighting ablation counterpart. Same situation.

  typescript  /  MagicMirror/typescript
      Both are git-TRACKED, so deleting shows as a repo change. Tier 1 removes
      only the MagicMirror/ copy. Delete the root one too if you want it gone
      from the project entirely.

  GESTURE_IMPROVEMENTS.md, SESSION_PROGRESS_2026-03-31.md, PLAN.md
      Older planning notes, untracked or stale. Not junk — your own writing.
      Left alone.

  camera/ws_bridge.py                                         6 KB
      WebSocket transport to the MagicMirror frontend. main.py uses
      http_sender.py instead, so it is currently unused — but it is a working
      alternative transport, not a superseded file. Delete only if you are
      certain the HTTP path is permanent.

  camera/export_dataset.py
      rsync transfer to/from the Mac. Unused now that recording is local, but
      it is the tool for the next cross-machine move.
REVIEW

echo
echo "───────────────────────────────────────────────────────────────────────"
(( APPLY )) && echo "Tier 1 freed: $(numfmt --to=iec "$freed")" \
            || echo "Tier 1 would free: $(numfmt --to=iec "$freed")  (re-run with --apply)"
echo "Tier 2 kept in full — this script never deletes it."

cat <<'NOTE'

NOT DELETED — identical content, but each copy is required
  MagicMirror/.husky/_/*                    14 hook shims; git calls them by name
  modules/MMM-{page-indicator,pages}/LICENSE.md   separate third-party modules
  modules/default/alert/translations/pt{,-br}.json  distinct locales
  tests/mocks/calendar_test{,_clone}.ics     fixtures referenced by name
  modules/*/.github/{dependabot.yaml,ISSUE_TEMPLATE/config.yml}

Similar names, different content — NOT duplicates
  camera/dataset/Chanakya/*_1.jpg            separate face captures
  tests/mocks/calendar_duplicates_{1,2}.ics  distinct fixtures
NOTE

cat <<'KEPT'

KEPT despite nothing importing them — these are ENTRY POINTS, not dead code
  gesture_system/train_twostream.py     the current trainer
  gesture_system/experiments.py         produced the report figures
  gesture_system/scaling_study.py       produced the scaling curves
  gesture_system/test_pipeline.py       24 regression tests on the encoding
  gesture_system/audit_null.py          re-run whenever null clips are added
  gesture_system/review_clips.py        32 saved verdicts came from it
  gesture_system/make_demo_reel.py      made the demo video
  gesture_system/render_pose_video.py   imported by make_demo_reel.py
  generate_variants.py, animate_variants.py, validate_generated.py,
  compare_generated.py                  dormant until server access
KEPT
