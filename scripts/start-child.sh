#!/usr/bin/env bash
# start-child.sh — Start MagicMirror + camera pipeline with the child profile.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MM_CONFIG_FILE="config/config.child.js" "$ROOT/scripts/start.sh"
