/**
 * MMM-CameraBridge node_helper
 * Runs a local HTTP server that the camera pipeline POSTs events to.
 */

const NodeHelper = require("node_helper");
const http = require("http");

module.exports = NodeHelper.create({

  start() {
    this.server = null;
    this.debugState = {
      updatedAt: null,
      presence: "away",
      fps: 0,
      gesture: null,
      fingerState: null,
      fingerLabel: null,
      faces: [],
      bestProfile: "unknown",
      bestConfidence: 0,
      model: {
        faceModelLoaded: false,
        knownProfiles: 0,
        gestureEnabled: false
      }
    };
    this.latestFrame = null;
  },

  socketNotificationReceived(notification, payload) {
    if (notification === "START_SERVER") {
      this.startServer(payload.port);
    }
  },

  startServer(port) {
    if (this.server) return;

    this.server = http.createServer((req, res) => {
      const urlPath = (req.url || "").split("?")[0];

      if (req.method === "GET" && (urlPath === "/" || urlPath === "/health")) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, service: "MMM-CameraBridge" }));
        return;
      }

      if (req.method === "GET" && urlPath === "/camera-debug") {
        const payload = {
          ok: true,
          debug: {
            ...this.debugState,
            hasFrame: Boolean(this.latestFrame)
          }
        };
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(payload));
        return;
      }

      if (req.method === "GET" && urlPath === "/camera-frame.jpg") {
        if (!this.latestFrame) {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "no frame yet" }));
          return;
        }
        res.writeHead(200, {
          "Content-Type": "image/jpeg",
          "Cache-Control": "no-store"
        });
        res.end(this.latestFrame);
        return;
      }

      if (req.method === "GET" && urlPath === "/camera-view") {
        const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Camera Debug</title>
  <style>
    body { font-family: monospace; background: #111; color: #eee; margin: 0; padding: 16px; }
    .wrap { display: grid; grid-template-columns: 640px 1fr; gap: 16px; }
    img { width: 640px; height: 480px; background: #222; border: 1px solid #444; object-fit: cover; }
    pre { background: #1b1b1b; border: 1px solid #333; padding: 12px; min-height: 456px; overflow: auto; }
    h1 { margin: 0 0 12px; font-size: 18px; }
    .muted { color: #aaa; font-size: 12px; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>MMM-CameraBridge Debug View</h1>
  <div class="wrap">
    <img id="frame" src="/camera-frame.jpg" alt="camera frame">
    <pre id="state">loading...</pre>
  </div>
  <div class="muted">Endpoints: /camera-debug, /camera-frame.jpg, /camera-event, /camera-debug (POST)</div>
  <script>
    const frame = document.getElementById("frame");
    const state = document.getElementById("state");

    async function tick() {
      try {
        const r = await fetch("/camera-debug", { cache: "no-store" });
        const j = await r.json();
        state.textContent = JSON.stringify(j, null, 2);
        frame.src = "/camera-frame.jpg?t=" + Date.now();
      } catch (e) {
        state.textContent = "debug fetch failed: " + e;
      }
    }

    tick();
    setInterval(tick, 500);
  </script>
</body>
</html>`;
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(html);
        return;
      }

      if (req.method !== "POST" || (urlPath !== "/camera-event" && urlPath !== "/camera-debug")) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "not found" }));
        return;
      }

      let body = "";
      req.on("data", chunk => { body += chunk; });
      req.on("end", () => {
        try {
          const payload = JSON.parse(body);
          if (urlPath === "/camera-event") {
            this.sendSocketNotification("CAMERA_EVENT", payload);
          } else {
            this.updateDebugState(payload);
          }
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        } catch (e) {
          console.error("MMM-CameraBridge: bad JSON:", body);
          res.writeHead(400);
          res.end(JSON.stringify({ error: "invalid JSON" }));
        }
      });
    });

    this.server.listen(port, "127.0.0.1", () => {
      console.log(`MMM-CameraBridge: listening on http://127.0.0.1:${port}/camera-event`);
      console.log(`MMM-CameraBridge: debug view http://127.0.0.1:${port}/camera-view`);
    });

    this.server.on("error", err => {
      console.error("MMM-CameraBridge: server error:", err.message);
    });
  },

  updateDebugState(payload) {
    const frameB64 = payload.frameJpegBase64;
    if (frameB64) {
      try {
        this.latestFrame = Buffer.from(frameB64, "base64");
      } catch (e) {
        console.error("MMM-CameraBridge: invalid debug frame payload");
      }
    }

    this.debugState = {
      updatedAt: payload.updatedAt || new Date().toISOString(),
      presence: payload.presence || "away",
      fps: Number(payload.fps || 0),
      gesture: payload.gesture || null,
      fingerState: payload.fingerState || null,
      fingerLabel: payload.fingerLabel || null,
      faces: Array.isArray(payload.faces) ? payload.faces : [],
      bestProfile: payload.bestProfile || "unknown",
      bestConfidence: Number(payload.bestConfidence || 0),
      model: payload.model || this.debugState.model
    };
  },

  stop() {
    if (this.server) {
      this.server.close();
      this.server = null;
    }
  },
});
