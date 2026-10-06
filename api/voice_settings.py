"""Builds the Deepgram Voice Agent `Settings` message.

Protocol facts (checked against developers.deepgram.com):
  - WebSocket: wss://agent.deepgram.com/v1/agent/converse
  - First client message after `Welcome` is {"type": "Settings", ...};
    wait for `SettingsApplied` before streaming audio.
  - Custom LLM: agent.think.provider.type = "open_ai" plus a SIBLING
    agent.think.endpoint = {"url", "headers"} (not fields inside
    provider). The endpoint must act like OpenAI Chat Completions.
  - provider.model is required by the schema but is just a label for us:
    api/voice.py ignores it and uses the generation layer's own model.

Configuration (env; all optional except the public URL):
  PUBLIC_BASE_URL          https URL Deepgram's cloud can reach (ngrok in dev)
  THINK_ENDPOINT_SECRET    if set, sent as a Bearer header and enforced by api/voice.py
  DEEPGRAM_LISTEN_MODEL    default nova-3
  DEEPGRAM_TTS_MODEL       default aura-2-thalia-en
  VOICE_GREETING           spoken first

Print the JSON:  python3 -m api.voice_settings
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_AGENT_WS_URL = "wss://agent.deepgram.com/v1/agent/converse"
THINK_PATH = "/api/voice/v1/chat/completions"
DEFAULT_GREETING = "Hi, I'm the HR assistant. What would you like to know about company policy?"


def build_agent_settings(
    public_base_url: str | None = None,
    *,
    listen_model: str | None = None,
    tts_model: str | None = None,
    greeting: str | None = None,
    think_secret: str | None = None,
) -> dict:
    base = (public_base_url or os.environ.get("PUBLIC_BASE_URL", "")).rstrip("/")
    if not base.startswith("https://"):
        raise ValueError(
            "Deepgram's cloud must be able to reach the think endpoint: set PUBLIC_BASE_URL "
            "(or pass public_base_url) to a public https URL, e.g. an ngrok tunnel. "
            "localhost will not work - see api/README.md."
        )
    secret = think_secret if think_secret is not None else os.environ.get("THINK_ENDPOINT_SECRET")
    headers = {"authorization": f"Bearer {secret}"} if secret else {}

    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 24000},
            "output": {"encoding": "linear16", "sample_rate": 24000, "container": "none"},
        },
        "agent": {
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": listen_model or os.environ.get("DEEPGRAM_LISTEN_MODEL", "nova-3"),
                }
            },
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "endpoint": {"url": base + THINK_PATH, "headers": headers},
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": tts_model or os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
                }
            },
            "greeting": greeting or os.environ.get("VOICE_GREETING", DEFAULT_GREETING),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_agent_settings(), indent=2))
