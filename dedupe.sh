#!/usr/bin/env bash
#
# dedupe.sh — remove verified redundant files under Magicmirror3/
#
# DRY RUN BY DEFAULT. Nothing is deleted until you run:  ./dedupe.sh --apply
#
# Scope of the scan that produced this list:
#   3427 files hashed (sha256), recursive, excluding
#   .git/ node_modules/ .venv/ venv/ site-packages/ dist-info/ __pycache__/ archive/
#
# It found 10 groups of byte-identical files. SEVEN of them are legitimate and
# are NOT touched here — see "DELIBERATELY NOT DELETED" at the bottom. Only the
# three groups below are genuinely redundant.
#
# Recoverable: ~6.9 MB, almost all of it one duplicated model file.

set -uo pipefail
cd "$(dirname "$0")" || exit 1

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

if [[ $APPLY -eq 0 ]]; then
  echo "DRY RUN — nothing will be deleted. Re-run with --apply to act."
else
  echo "APPLYING — files listed below will be deleted."
fi
echo

freed=0

# verify() re-checks the hashes match RIGHT NOW before deleting anything.
# The scan was a snapshot; if either file changed since, this refuses to act.
drop_if_identical() {
  local keep="$1" drop="$2" why="$3"

  if [[ ! -f "$keep" || ! -f "$drop" ]]; then
    echo "SKIP  missing file, nothing to do"
    echo "      keep: $keep"
    echo "      drop: $drop"
    echo
    return
  fi

  local hk hd
  hk=$(sha256sum "$keep" | cut -d' ' -f1)
  hd=$(sha256sum "$drop" | cut -d' ' -f1)

  if [[ "$hk" != "$hd" ]]; then
    echo "SKIP  contents differ now — NOT deleting"
    echo "      $keep  $hk"
    echo "      $drop  $hd"
    echo
    return
  fi

  local sz
  sz=$(stat -c%s "$drop")
  echo "DUPE  $why"
  echo "      keep   $keep"
  echo "      delete $drop  ($(numfmt --to=iec "$sz"))"
  if [[ $APPLY -eq 1 ]]; then
    rm -- "$drop" && echo "      -> deleted"
  fi
  freed=$((freed + sz))
  echo
}

# ---------------------------------------------------------------------------
# 1. YOLO pose model, stored twice (6.8 MB each)
#    No code in the repo references either path, so neither is load-bearing by
#    filename. Keeping the copy under camera/ since that is where the vision
#    code lives; MagicMirror/ is the Node frontend.
#    NOTE: both are git-tracked, so this shows up as a deletion in git status.
# ---------------------------------------------------------------------------
drop_if_identical \
  "camera/yolov8n-pose.pt" \
  "MagicMirror/yolov8n-pose.pt" \
  "duplicated YOLO pose weights"

# ---------------------------------------------------------------------------
# 2. `typescript` — output of the `script` command (a captured terminal session
#    from 2026-03-27), committed twice by accident. Not source code.
#    NOTE: both are git-tracked.
# ---------------------------------------------------------------------------
drop_if_identical \
  "typescript" \
  "MagicMirror/typescript" \
  "stray 'script' terminal capture, committed twice"

# ---------------------------------------------------------------------------
# 3. magicmirror.log — identical copies, untracked, and .gitignore already
#    covers *.log and logs/. Keeping the one inside MagicMirror/ since that is
#    where the app writes.
# ---------------------------------------------------------------------------
drop_if_identical \
  "MagicMirror/logs/magicmirror.log" \
  "logs/magicmirror.log" \
  "duplicated runtime log"

echo "-----------------------------------------------------------"
if [[ $APPLY -eq 1 ]]; then
  echo "Freed: $(numfmt --to=iec "$freed")"
else
  echo "Would free: $(numfmt --to=iec "$freed")   (re-run with --apply)"
fi
cat <<'NOTE'

DELIBERATELY NOT DELETED
------------------------
These are byte-identical but each copy is required. Removing them breaks things.

  MagicMirror/.husky/_/{pre-commit,post-merge,...}      14 identical files
      Husky generates one shim per git hook. Git invokes them by filename.

  modules/MMM-page-indicator/LICENSE.md
  modules/MMM-pages/LICENSE.md
      Two separate third-party modules that happen to ship the same MIT text.
      Each needs its own licence file.

  modules/default/alert/translations/pt.json
  modules/default/alert/translations/pt-br.json
      Portuguese and Brazilian Portuguese. Identical today, looked up by
      filename, and they can diverge later.

  tests/mocks/calendar_test.ics
  tests/mocks/calendar_test_clone.ics
  tests/configs/modules/calendar/{berlin_end_of_day_repeating,end_of_day_berlin_moved}.js
      Test fixtures referenced by name. "clone" is intentional.

  modules/*/.github/dependabot.yaml
  modules/*/.github/ISSUE_TEMPLATE/config.yml
      Per-module third-party config. Same content, different repos.

Also checked and NOT duplicates (similar names, different content):
  camera/dataset/Chanakya/*_1.jpg          separate face captures
  tests/mocks/calendar_duplicates_{1,2}.ics  distinct fixtures
NOTE
