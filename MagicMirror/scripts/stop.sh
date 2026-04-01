#!/usr/bin/env bash
# stop.sh — Stop MagicMirror + camera pipeline
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$ROOT/logs"

stop_pid() {
  local name=$1
  local pidfile="$LOGS_DIR/${name}.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "[stop] Stopped $name (PID $pid)"
    else
      echo "[stop] $name was not running"
    fi
    rm -f "$pidfile"
  else
    echo "[stop] No PID file for $name"
  fi
}

stop_pid "magicmirror"
stop_pid "camera"
