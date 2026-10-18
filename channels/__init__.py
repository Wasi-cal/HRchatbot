"""Text and voice are fully separate pipelines, one module each:

    channels/text.py   /api/chat      retrieval -> grounding -> generation, small talk, copy
    channels/voice.py  /api/voice     the same flow for the spoken channel

They share only primitives that have no channel behaviour (channels/common.py,
vectorstore, DB/cache, the LLM client). Neither module imports the other
(enforced by eval/channel_separation_test.py), so a bug in one channel's
flow, copy or settings can be debugged without reading the other's.
"""
import importlib

CHANNELS = ("text", "voice")


def get_channel(name: str):
    """The pipeline module for a channel. Used only by compatibility routers
    (generation.generate.generate_answer, access_control.ask_and_answer)."""
    if name not in CHANNELS:
        raise ValueError(f"unknown channel {name!r}; expected one of {CHANNELS}")
    return importlib.import_module(f"channels.{name}")


# One INFO line per turn from "hrchatbot.text" / "hrchatbot.voice" (turn type, prompt versions,
# timings), so a channel's behaviour can be followed with a simple grep. uvicorn does not
# configure application loggers, so attach a handler here if none exists yet.
import logging

_root = logging.getLogger("hrchatbot")
if not _root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    _root.addHandler(_handler)
    _root.setLevel(logging.INFO)
    _root.propagate = False
