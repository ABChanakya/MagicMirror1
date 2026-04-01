"""
http_sender.py
Posts camera events to MMM-CameraBridge's local HTTP server.
"""

import json
import logging
import requests

logger = logging.getLogger(__name__)


class HttpSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 8081):
        self.url = f"http://{host}:{port}/camera-event"

    def send(self, event: dict) -> bool:
        """POST a single event dict. Returns True on success."""
        try:
            resp = requests.post(self.url, json=event, timeout=1.0)
            resp.raise_for_status()
            return True
        except requests.exceptions.ConnectionError:
            logger.warning("CameraBridge not reachable at %s — is MagicMirror running?", self.url)
            return False
        except Exception as e:
            logger.error("Failed to send event %s: %s", event, e)
            return False

    def send_gesture(self, name: str):
        return self.send({"type": "gesture", "name": name})

    def send_face(self, profile: str, confidence: float = 1.0):
        return self.send({"type": "face", "profile": profile, "confidence": round(confidence, 3)})

    def send_presence(self, state: str):
        """state: 'present' or 'away'"""
        return self.send({"type": "presence", "state": state})
