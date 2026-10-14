#!/usr/bin/env python3
"""Proves text and voice are fully separate: code, settings, prompts and copy.

    python3 -m eval.channel_separation_test

Needs the same live services as eval/run_regression.py. Every setting or prompt it
touches is restored at the end (even on failure).
"""
import ast
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import channels.text as text_ch  # noqa: E402
import channels.voice as voice_ch  # noqa: E402
import config.prompts as prompts_mod  # noqa: E402
from config.runtime import get_app_config, set_app_config  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402
from vectorstore.retrieve import retrieve  # noqa: E402

NOTE = "channel_separation_test"
_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))


def imports_of(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_code():
    print("\n=== (a) Code separation ===")
    t, v = imports_of(ROOT / "channels/text.py"), imports_of(ROOT / "channels/voice.py")
    check("channels/text.py never imports the voice channel", not any("voice" in n.split(".")[-1] or n == "channels.voice" for n in t if n.startswith("channels")), str(sorted(t)))
    check("channels/voice.py never imports the text channel", not any(n == "channels.text" or n.endswith(".text") for n in v if n.startswith("channels")), str(sorted(v)))
    check("api/chat.py uses only the text channel", "channels.voice" not in imports_of(ROOT / "api/chat.py")
          and "voice" not in {n.split(".")[-1] for n in imports_of(ROOT / "api/chat.py") if n.startswith("channels")})
    voice_api = imports_of(ROOT / "api/voice.py")
    check("api/voice.py uses only the voice channel", "channels.text" not in voice_api and "channels.voice" in voice_api or "channels.voice" in " ".join(voice_api))
    check("each channel has its own copy, patterns and replies (distinct objects)",
          text_ch._RESPONSES is not voice_ch._RESPONSES
          and text_ch._matcher is not voice_ch._matcher and text_ch._picker is not voice_ch._picker)
    check("each channel logs under its own logger", (text_ch.logger.name, voice_ch.logger.name) == ("hrchatbot.text", "hrchatbot.voice"))
    check("phrase sets are separate app_config rows per channel (smalltalk.phrases.text / .voice)",
          get_app_config("smalltalk.phrases.text") != get_app_config("smalltalk.phrases.voice"))
    check("voice-only phrasing is voice-only ('can you hear me')",
          voice_ch.smalltalk_match("can you hear me") is not None and text_ch.smalltalk_match("can you hear me") is None)
    check("no-match copy is channel-specific", text_ch.NO_MATCH_BASE != voice_ch.NO_MATCH_BASE)


def test_settings():
    print("\n=== (b) Settings: editing one channel never changes the other ===")
    q = "how much notice do I need to serve"
    orig = {k: get_app_config(k) for k in ("retrieval.top_k.text", "retrieval.top_k.voice", "retrieval.candidate_n.text",
                                           "retrieval.candidate_n.voice", "generation.confidence_threshold.voice",
                                           "generation.confidence_threshold.text")}
    try:
        set_app_config("retrieval.top_k.voice", 1)
        check("voice top_k=1 -> voice retrieval returns 1, text unchanged",
              len(retrieve(q, channel="voice")) == 1 and len(retrieve(q, channel="text")) == orig["retrieval.top_k.text"])
        set_app_config("retrieval.top_k.voice", orig["retrieval.top_k.voice"])
        set_app_config("retrieval.top_k.text", 2)
        check("text top_k=2 -> text retrieval returns 2, voice unchanged",
              len(retrieve(q, channel="text")) == 2 and len(retrieve(q, channel="voice")) == orig["retrieval.top_k.voice"])
        set_app_config("retrieval.top_k.text", orig["retrieval.top_k.text"])

        # confidence threshold: voice 0.99 -> voice refuses everything; text still answers
        set_app_config("generation.confidence_threshold.voice", 0.99)
        v_ev = list(voice_ch.answer_stream("how many days of earned leave do I get"))[-1]
        t_ev = list(text_ch.answer_stream("how many days of earned leave do I get"))[-1]
        check("voice threshold 0.99: voice takes the retrieval shortcut", v_ev["shortcut"] is True)
        check("...while text, with its own threshold, still generates an answer", t_ev["shortcut"] is False)
        set_app_config("generation.confidence_threshold.voice", orig["generation.confidence_threshold.voice"])
        set_app_config("generation.confidence_threshold.text", 0.99)
        v_ev = list(voice_ch.answer_stream("how many days of earned leave do I get"))[-1]
        t_ev = list(text_ch.answer_stream("how many days of earned leave do I get"))[-1]
        check("text threshold 0.99: text refuses, voice still answers", t_ev["shortcut"] is True and v_ev["shortcut"] is False)
    finally:
        for k, v in orig.items():
            set_app_config(k, v)

    check("every channel setting exists for both channels",
          all(get_app_config(f"{k}.{c}") is not None
              for k in ("retrieval.top_k", "retrieval.candidate_n", "retrieval.rrf_k", "generation.model",
                        "generation.confidence_threshold")
              for c in ("text", "voice")))


def test_prompts():
    print("\n=== (c) Prompts: each channel has independent grounding AND style ===")
    from api.main import app
    from fastapi.testclient import TestClient

    prompts_mod.refresh_prompt_cache()  # start from DB truth, not a cache entry up to 30s old
    before_text, before_voice = prompts_mod.get_system_prompt("text"), prompts_mod.get_system_prompt("voice")
    old_active = {}
    with get_pool().connection() as conn:
        for ch in ("text", "voice"):
            old_active[ch] = conn.execute(
                "SELECT id, content FROM prompt_templates WHERE name='grounding_core' AND channel=%s AND is_active", (ch,)).fetchone()
    marker = "VOICEONLY-GROUNDING-MARKER"
    with get_pool().connection() as conn:  # a new version is MAX(version)+1, which can exceed the active one when a candidate version exists inactive
        max_voice_version = conn.execute("SELECT MAX(version) FROM prompt_templates WHERE name='grounding_core' AND channel='voice'").fetchone()[0]
    try:
        with TestClient(app) as client:
            r = client.post("/api/admin/prompts", json={"name": "grounding_core", "channel": "voice",
                                                        "content": old_active["voice"][1] + f"\n- {marker}", "note": NOTE})
            check("new grounding_core/voice version created", r.status_code == 200, r.text[:200])
        t_now, v_now = prompts_mod.get_system_prompt("text"), prompts_mod.get_system_prompt("voice")
        check("voice prompt picked up its new grounding version immediately",
              marker in v_now[0] and v_now[1]["grounding_core"] == max_voice_version + 1, str(v_now[1]))
        check("text prompt and its version_info are untouched", t_now == before_text, str(t_now[1]))
    finally:
        with get_pool().connection() as conn:
            # delete the test version first (it is the active one), then re-activate the original
            conn.execute("DELETE FROM prompt_templates WHERE created_by = %s", (NOTE,))
            for ch in ("text", "voice"):
                conn.execute("UPDATE prompt_templates SET is_active = true WHERE id = %s", (old_active[ch][0],))
            conn.commit()
        prompts_mod.refresh_prompt_cache()
    check("prompts restored", prompts_mod.get_system_prompt("voice") == before_voice and prompts_mod.get_system_prompt("text") == before_text)


def test_compat_routers():
    print("\n=== (d) Compatibility routers dispatch to the channel modules ===")
    from access_control.ask_and_answer import ask_and_answer
    from generation.generate import generate_answer
    for ch in ("text", "voice"):
        done = [e for e in generate_answer("how many days of earned leave do I get", channel=ch) if e["type"] == "done"][0]
        r = ask_and_answer("how many days of earned leave do I get", channel=ch)
        check(f"generate_answer / ask_and_answer route channel={ch!r}",
              done["prompt_version"] == prompts_mod.get_system_prompt(ch)[1] and r["type"] == "answer"
              and r["prompt_version"] == prompts_mod.get_system_prompt(ch)[1], str(done["prompt_version"]))
    try:
        ask_and_answer("x", None, channel="fax")
        check("unknown channel rejected", False)
    except ValueError:
        check("unknown channel rejected", True)


def main():
    try:
        test_code()
        test_settings()
        test_prompts()
        test_compat_routers()
    finally:
        close_pool()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
