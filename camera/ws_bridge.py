"""
ws_bridge.py — WebSocket broadcast server for the MagicMirror3 camera pipeline.

Runs an asyncio WebSocket server in a daemon thread so it never blocks the
main camera loop. Broadcasts JSON events to all connected MagicMirror clients.

Requires: websockets>=8.0,<10  (Python 3.6 compatible)
  pip3 install 'websockets>=8.0,<10'

Python 3.6+ compatible — no walrus, no str|None, no match/case.
"""

import asyncio
import json
import logging
import threading
from typing import Optional, Set

logger = logging.getLogger(__name__)


class WsBridge(object):
    """
    Thread-safe WebSocket broadcaster.

    Start once at pipeline startup, then call broadcast() from any thread.
    If websockets is not installed, or the port is in use, start() logs a
    warning and broadcast() becomes a silent no-op — the camera loop keeps
    running regardless.
    """

    def __init__(self, host="0.0.0.0", port=8084):
        self.host = host
        self.port = port
        self._clients = set()   # type: Set
        self._loop = None       # type: Optional[asyncio.AbstractEventLoop]
        self._thread = None     # type: Optional[threading.Thread]
        self._ready = threading.Event()

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self):
        """Start the WebSocket server in a daemon thread. Blocks until ready (max 5s)."""
        self._thread = threading.Thread(target=self._run, name="ws-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            logger.warning("WsBridge: server did not start within 5 s")

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def broadcast(self, event):
        # type: (dict) -> None
        """Broadcast a JSON-serialisable event dict to all connected clients.
        Safe to call from any thread. No-op if no clients are connected."""
        if self._loop is None or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(event), self._loop)

    def client_count(self):
        # type: () -> int
        return len(self._clients)

    # ── Internal ────────────────────────────────────────────────────────────

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            logger.error("WsBridge: event loop crashed: %s", exc)
        finally:
            self._ready.set()   # unblock start() if still waiting

    async def _serve(self):
        try:
            import websockets
        except ImportError:
            logger.error(
                "WsBridge: 'websockets' not installed. "
                "Run: pip3 install 'websockets>=8.0,<10'"
            )
            self._ready.set()
            return

        try:
            # async-context-manager style works in websockets 8.x and 9.x
            async with websockets.serve(self._handler, self.host, self.port):
                logger.info("WsBridge: ws://%s:%d ready", self.host, self.port)
                self._ready.set()
                await asyncio.Future()          # run until loop is stopped
        except OSError as exc:
            logger.error("WsBridge: cannot bind %s:%d — %s", self.host, self.port, exc)
            self._ready.set()
        except Exception as exc:
            logger.error("WsBridge: serve error: %s", exc)
            self._ready.set()

    async def _handler(self, websocket, path=""):
        # path kwarg keeps compatibility with websockets 8.x (which passes path)
        remote = getattr(websocket, "remote_address", "?")
        self._clients.add(websocket)
        logger.info("WsBridge: client connected %s (%d total)", remote, len(self._clients))
        try:
            await websocket.wait_closed()
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("WsBridge: client gone %s (%d remaining)", remote, len(self._clients))

    async def _broadcast_async(self, event):
        if not self._clients:
            return
        data = json.dumps(event)
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception as exc:
                logger.debug("WsBridge: send failed (%s), dropping client", exc)
                dead.add(ws)
        if dead:
            self._clients -= dead
