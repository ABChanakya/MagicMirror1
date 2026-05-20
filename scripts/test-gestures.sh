#!/bin/bash
# test-gestures.sh — Demo gesture simulator for local testing
# Sends HTTP POST requests to simulate camera events
# Usage: bash scripts/test-gestures.sh
# Then interact with the gesture panel at http://localhost:8081

BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8082}"
DELAY="${DELAY:-0.5}"

echo "🎮 MagicMirror3 Gesture Simulator"
echo "=================================="
echo "Bridge URL: $BRIDGE_URL"
echo "Press Ctrl+C to stop"
echo ""

send_gesture() {
  local gesture=$1
  local label=$2
  echo -n "Sending $label... "
  curl -s -X POST "$BRIDGE_URL/camera-event" \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"gesture\",\"name\":\"$gesture\"}" > /dev/null 2>&1
  echo "✓"
  sleep "$DELAY"
}

send_presence() {
  local state=$1
  echo -n "Sending presence: $state... "
  curl -s -X POST "$BRIDGE_URL/camera-event" \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"presence\",\"state\":\"$state\"}" > /dev/null 2>&1
  echo "✓"
  sleep "$DELAY"
}

send_greeting() {
  local name=$1
  echo -n "Sending greeting for $name... "
  curl -s -X POST "$BRIDGE_URL/camera-event" \
    -H "Content-Type: application/json" \
    -d "{\"type\":\"greeting\",\"names\":[\"$name\"]}" > /dev/null 2>&1
  echo "✓"
  sleep "$DELAY"
}

# Menu
while true; do
  echo ""
  echo "Choose gesture to test:"
  echo "  1) Finger gestures (1-5)"
  echo "  2) Fist (home)"
  echo "  3) Motion gestures (swipe)"
  echo "  4) Presence detection"
  echo "  5) Greeting"
  echo "  6) Auto demo (all gestures)"
  echo "  Q) Quit"
  echo ""
  read -p "Enter choice: " choice

  case "$choice" in
    1)
      echo ""
      for i in 1 2 3 4 5; do
        send_gesture "fingers_$i" "fingers_$i"
      done
      ;;
    2)
      echo ""
      send_gesture "fist" "fist"
      ;;
    3)
      echo ""
      send_gesture "swipe_up" "swipe up"
      send_gesture "swipe_down" "swipe down"
      send_gesture "swipe_left" "swipe left"
      send_gesture "swipe_right" "swipe right"
      ;;
    4)
      echo ""
      send_presence "present"
      sleep 2
      send_presence "away"
      ;;
    5)
      echo ""
      read -p "Enter name: " name
      send_greeting "${name:-Chanakya}"
      ;;
    6)
      echo ""
      echo "Running auto demo..."
      echo ""
      send_presence "present"
      send_greeting "Demo User"
      sleep 1
      send_gesture "fingers_1" "fingers_1 (page 1)"
      sleep 1
      send_gesture "fingers_2" "fingers_2 (page 2)"
      sleep 1
      send_gesture "fingers_5" "fingers_5 (flip card)"
      sleep 1
      send_gesture "swipe_up" "swipe_up"
      sleep 1
      send_gesture "fist" "fist (home)"
      sleep 1
      send_presence "away"
      echo ""
      echo "Demo complete! ✅"
      ;;
    [Qq])
      echo "Bye! 👋"
      exit 0
      ;;
    *)
      echo "Invalid choice"
      ;;
  esac
done
