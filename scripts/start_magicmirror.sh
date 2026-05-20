#!/bin/bash
# start_magicmirror.sh — Start MagicMirror + camera pipeline on Jetson Nano
# Install: cp scripts/start_magicmirror.sh ~/start_magicmirror.sh && chmod +x ~/start_magicmirror.sh
# Run: bash ~/start_magicmirror.sh

set -euo pipefail

# ── Swap check ────────────────────────────────────────────────────────────
if [ "$(swapon --show | wc -l)" -eq 0 ]; then
  echo "⚠️  No swap enabled. Creating 4GB swapfile..."
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
  echo "✅ Swap configured"
fi

# ── Max performance mode ──────────────────────────────────────────────────
echo "🚀 Setting Jetson to max performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# ── Create logs directory ─────────────────────────────────────────────────
LOGS_DIR="$HOME/logs"
mkdir -p "$LOGS_DIR"

# ── Start Docker container ────────────────────────────────────────────────
echo "🐳 Starting MagicMirror Docker container..."
docker start magicmirror 2>/dev/null || docker run -d \
  --name magicmirror \
  --restart unless-stopped \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  mymagicmirror:latest
sleep 8

# ── Start camera pipeline ────────────────────────────────────────────────
echo "📷 Starting camera pipeline..."
export DISPLAY=:0
export XAUTHORITY="$HOME/.Xauthority"

cd "$HOME/MagicMirror1/camera"
nohup bash run.sh \
  --device /dev/video0 \
  --bridge-port 8082 \
  > "$LOGS_DIR/camera.log" 2>&1 &
CAMERA_PID=$!
echo "$CAMERA_PID" > "$LOGS_DIR/camera.pid"
echo "✅ Camera PID: $CAMERA_PID"

sleep 5

# ── Open browser in kiosk mode ──────────────────────────────────────────
echo "🌐 Opening MagicMirror in browser..."
DISPLAY=:0 chromium-browser \
  --kiosk \
  --no-sandbox \
  --disable-gpu \
  --disable-software-rasterizer \
  --disable-dev-shm-usage \
  http://localhost:8081 > /dev/null 2>&1 &
BROWSER_PID=$!
echo "✅ Browser PID: $BROWSER_PID"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✨ MagicMirror3 is running!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 Monitor:"
echo "   tail -f ~/logs/camera.log"
echo ""
echo "🛑 Stop:"
echo "   pkill -f 'python3 main.py' && docker stop magicmirror"
echo ""
echo "🔄 Restart:"
echo "   bash ~/start_magicmirror.sh"
echo ""
