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

  - agent.listen.provider (docs: configure-voice-agent, voice-agent-stt-models):
      model    nova-3 (version v1, the default) or flux-general-en (Flux: version "v2" is REQUIRED)
      keyterms list of plain strings, no weights, a multi-word phrase is ONE element; limited to 500 tokens per request
               (works on Nova-3 and Flux; the docs do not say whether the list can change after session start - it is
               sent in the Settings message, i.e. at session start)
      eot_threshold       0.5-1.0 (default 0.7), Flux only; 1.0 would suppress natural end-of-turn
      eager_eot_threshold 0.3-0.9, Flux only, must be <= eot_threshold - NEVER sent here: eager mode "increases LLM
               calls" (speculative generation, TurnResumed cancellations), so it stays off
      eot_timeout_ms      500-60000 (default 5000), Flux only, optional
    The eot_* fields are only sent for Flux models.

Configuration:
  env  PUBLIC_BASE_URL        https URL Deepgram's cloud can reach (ngrok in dev)
  env  THINK_ENDPOINT_SECRET  if set, sent as a Bearer header and enforced by api/voice.py
  DB   provider_configs stt.model|version|keyterms|eot_threshold|eager_eot_threshold|eot_timeout_ms, tts.voice,
       tts.greeting (editable via /api/admin/config; seeded once from DEEPGRAM_LISTEN_MODEL, DEEPGRAM_TTS_MODEL,
       VOICE_GREETING). Explicit function args override.

Print the JSON:  python3 -m api.voice_settings
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from config.runtime import (
    DEFAULT_GREETING,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_VOICE,
    get_active_provider_config,
)

load_dotenv()

DEEPGRAM_AGENT_WS_URL = "wss://agent.deepgram.com/v1/agent/converse"
ENV_PATH_DEFAULT = Path(__file__).resolve().parent.parent / ".env"
THINK_PATH = "/api/voice/v1/chat/completions"


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
    stt = get_active_provider_config("stt")
    tts = get_active_provider_config("tts")
    headers = {"authorization": f"Bearer {secret}"} if secret else {}

    model = listen_model or stt.get("model") or DEFAULT_STT_MODEL
    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 24000},
            "output": {"encoding": "linear16", "sample_rate": 24000, "container": "none"},
        },
        "agent": {
            "listen": {"provider": build_listen_provider(model, stt)},
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-4o-mini"},
                "endpoint": {"url": base + THINK_PATH, "headers": headers},
            },
            "speak": {
                "provider": {
                    "type": "deepgram",
                    "model": tts_model or tts.get("voice") or DEFAULT_TTS_VOICE,
                }
            },
            "greeting": greeting or tts.get("greeting") or DEFAULT_GREETING,
        },
    }


def build_listen_provider(model: str, stt: dict) -> dict:
    """agent.listen.provider from the stt row of provider_configs (see the module docstring for the field rules)."""
    provider = {"type": "deepgram", "model": model}
    flux = model.startswith("flux")
    if flux:
        provider["version"] = "v2"  # Deepgram: Flux requires v2; every other model uses v1 (the default, so omitted)
    keyterms = [t for t in (stt.get("keyterms") or []) if isinstance(t, str) and t.strip()]
    if keyterms:
        provider["keyterms"] = keyterms
    if flux:
        if stt.get("eot_threshold") is not None:
            provider["eot_threshold"] = stt["eot_threshold"]
        if stt.get("eager_eot_threshold") is not None:  # only if an admin set it deliberately; seeded unset
            provider["eager_eot_threshold"] = stt["eager_eot_threshold"]
        if stt.get("eot_timeout_ms") is not None:
            provider["eot_timeout_ms"] = stt["eot_timeout_ms"]
    return provider


if __name__ == "__main__":
    print(json.dumps(build_agent_settings(), indent=2))
