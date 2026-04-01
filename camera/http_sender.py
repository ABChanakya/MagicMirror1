"""
http_sender.py
Posts camera events to MMM-CameraBridge's local HTTP server.
"""

import json
import logging
import requests

logger = logging.getLogger(__name__)


class HttpSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 8082):
        self.url = f"http://{host}:{port}/camera-event"
        self.debug_url = f"http://{host}:{port}/camera-debug"

    def _post(self, url: str, payload: dict, timeout_s: float = 1.0) -> bool:
        try:
            resp = requests.post(url, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            logger.warning("CameraBridge not reachable at %s — is MagicMirror running?", url)
            return False
        except Exception as e:
            logger.error("Failed POST to %s: %s", url, e)
            return False

    def send(self, event: dict) -> bool:
        """POST a single event dict. Returns True on success."""
        return self._post(self.url, event, timeout_s=1.0)

    def send_gesture(self, name: str):
        return self.send({"type": "gesture", "name": name})

    def send_face(self, profile: str, confidence: float = 1.0):
        return self.send({"type": "face", "profile": profile, "confidence": round(confidence, 3)})

    def send_presence(self, state: str):
        """state: 'present' or 'away'"""
        return self.send({"type": "presence", "state": state})

    def send_debug(self, payload: dict):
        """POST debug telemetry (frame + model state) for tuning dashboard."""
        return self._post(self.debug_url, payload, timeout_s=0.8)
