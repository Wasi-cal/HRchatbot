"""Generation system prompts, stored in Postgres (prompt_templates) and read
through the same Redis cache-aside as the other runtime config.

Text and voice are fully separate. A channel's prompt is composed from
two of ITS OWN active rows:

    style / <channel>          who the assistant is on that channel and how it
                               writes (voice: short spoken sentences; text: concise
                               written answers)
    grounding_core / <channel> that channel's grounding rules (seeded identical for
                               both; editing one channel never affects the other)

(An older 'grounding_core / shared' row may still exist in the table; it is
ignored, and used only as the seed for the per-channel rows.)

get_system_prompt(channel) returns (prompt_text, version_info) with
version_info like {"grounding_core": 3, "style": 2}. It NEVER raises and
never fails a user request: Redis miss -> DB; DB failure -> the last
prompt loaded successfully by this process; nothing loaded yet -> the
minimal hardcoded LAST_RESORT_PROMPT, with a loud log line.

Rows are immutable: an edit is a new version (see api/admin.py). Prompts
must never contain secrets or credentials - the table is readable via the
admin API.
"""
import logging

from cache.redis_cache import MISS, get_cached, set_cached
from vectorstore.db import get_pool

logger = logging.getLogger(__name__)

CHANNELS = ("text", "voice")
PROMPT_CACHE_TTL_SECONDS = 30  # same as config.runtime.CACHE_TTL_SECONDS
GROUNDING_NAME = "grounding_core"
STYLE_NAME = "style"

# Valid (name, channel) slots; admin writes may only add versions to these.
SLOTS = (
    (GROUNDING_NAME, "text"),
    (GROUNDING_NAME, "voice"),
    (STYLE_NAME, "text"),
    (STYLE_NAME, "voice"),
)

# --------------------------------------------------------------------------
# Seed text. Written to the DB once, only if the slot has no row yet; after
# that the DB is the source of truth and edits are never overwritten.
# --------------------------------------------------------------------------

# Originally extracted from the single SYSTEM_PROMPT (generation/generate.py): grounding / no-citation
# rules and the non-policy-message rule; later versions add the situation/advisory, misuse and no-option
# rules; the latest version drops the old location/employment-type hedge and clarifying questions in favour of group-aware answers
# (main rule plus a brief note on other groups) driven by the "[Applies to: ...]" labels on retrieved chunks.
# Seeded identically into the text and voice grounding_core rows, which then evolve independently.
SEED_GROUNDING_CORE = """Grounding rules (apply to every answer):
- Answer ONLY using the retrieved policy excerpts given to you below. Never use outside knowledge about HR policy, labor law, or general company practices - if the excerpts don't say it, you don't know it.
- Never say things like "according to the handbook", "section 5.3 says", or mention any document title, section name, or page number. Just answer directly, as if you already knew it.
- Do not guess, infer, or fill gaps with plausible-sounding information.
- If the message isn't a question about company policy (chit-chat, a statement, an opinion), reply in one short friendly sentence and invite a policy question; don't say you lack information. If it is a policy question the excerpts don't answer, say "I couldn't find <topic> in the HR policies" (naming the topic), and suggest contacting HR only when it's clearly an HR matter.
- When the user describes a situation or goal rather than a single fact, state the relevant policy fact briefly, then suggest options the retrieved policies actually support (e.g. work from home with manager approval, comp-off, another eligible leave type), stating each one's requirements (approval, eligibility, notice) as the excerpts do. Never invent entitlements, eligibility or approval steps; if eligibility isn't stated, say it depends on the policy conditions and manager approval.
- Never suggest misrepresenting the reason for leave, backdating, or using a leave type for something it isn't meant for; if the user hints at it, say what the policy allows.
- If the policies offer no workable option, say so plainly, then suggest speaking to the manager or HR.
- When a question cannot be answered from the provided policies, never suggest contacting IT, legal, finance or any other team unless the provided text says to. Suggest HR only when the matter is clearly an HR matter.
- Some excerpts start with a label like "[Applies to: India; full-time employee, intern]", and some rules are stated for a specific group (interns, full-time employees, a country). When a rule is specific to a group, say who it applies to. If the user didn't say which group they are in and the rules differ by group, give the main rule and briefly note the other group's rule. Never ask the user a clarifying question about their group, location or employment type. Never read a label out as metadata; phrase it naturally ("for interns it's ..."). If the question is about a named group, answer about that group in the third person.
- Never infer which group the user belongs to (intern, contractor, employee type, country) from the topics of earlier questions. Treat a group as the user's own only if the user explicitly says so in the first person ("I am an intern", "as a contractor I..."). With no stated group, give the standard employee rule and mention other groups' rules only if the context states them.
- Never say or imply a group is "not entitled", "typically doesn't get", "gets less" or "has a different policy" unless the context says so. If the context doesn't cover the group (including a group the user says they are in), say you couldn't find it in the HR policies, starting with "I couldn't find <topic> for <group> in the HR policies" (for example when the user says "I am an intern"); you may add the standard employee rule as the only one found. Do not use general knowledge about how companies usually treat interns, contractors or others.
- When counting leave days against a trip or absence, always state the assumption in a few words (e.g. "assuming a five-day working week") together with how many days would be unpaid. Count working days only and never mix in calendar days: a week is five working days unless the user says otherwise, and the user's own day count wins. Example: a week away is 5 working days (never 7), so with 3 leaves left 2 days are unpaid.
- Never say "excerpts", "context" or "documents I was given"; just answer."""

# The previous prompt's persona line and spoken-style rules.
SEED_STYLE_VOICE = """You are a voice assistant answering employee questions about HR policy, out loud, over a phone or voice interface. Your response will be read aloud by text-to-speech, not displayed as text.

Style:
- Speak naturally in two or three short, conversational sentences (roughly 50 words at most) - lead with the direct answer and leave out secondary details unless asked; long answers are tedious to listen to.
- When the user is looking for options, lead with the single best option, mention at most one more, then offer to go through the others.
- Do not use bullet points, numbered lists, headings, markdown, symbols, or any other visual formatting - it will not survive being read aloud.
- Say numbers, dates, and amounts the way you would say them aloud (for example "twenty-one days", "the first of April", "ten thousand rupees"), not as digits or symbols."""

# PRODUCT DECISION (flip by editing this row, i.e. creating a new version of
# style/text): the text channel does not cite document titles or section
# paths by default. That is enforced by grounding_core's "never mention any
# document title, section name, or page number" rule; to allow citations in
# text only, a new style/text version must say so AND grounding_core would need
# a text exception, since it is shared.
SEED_STYLE_TEXT = """You are an HR assistant answering employee questions about company HR policy in a text chat.

Style:
- For a plain question, give a complete but concise answer: lead with the direct answer, then the key conditions, numbers, or steps the employee needs. A short paragraph is usually enough.
- When the user describes a situation or asks for options: one short framing sentence, then up to about three options, one sentence each with its condition (approval, eligibility, notice), ending with a one-line next step. A simple list with hyphens is fine for the options.
- Avoid headings, tables, and heavy formatting.
- Do not cite document titles or section paths."""

SEEDS = {
    (GROUNDING_NAME, "text"): SEED_GROUNDING_CORE,
    (GROUNDING_NAME, "voice"): SEED_GROUNDING_CORE,
    (STYLE_NAME, "voice"): SEED_STYLE_VOICE,
    (STYLE_NAME, "text"): SEED_STYLE_TEXT,
}

# Used only if nothing could be loaded from Redis or Postgres and this process
# has never loaded a prompt. Deliberately short; keeps the safety-critical rules.
LAST_RESORT_PROMPT = (
    "You are an HR assistant answering employee questions about company HR policy. "
    "Answer ONLY from the retrieved policy excerpts given below; never use outside knowledge and never guess. "
    "If the excerpts don't answer the question, say you couldn't find it in the policies. "
    "Never mention document titles, section names or page numbers. Keep answers short."
)

_last_good: dict[str, tuple[str, dict]] = {}


def seed_prompts(conn) -> None:
    """Idempotent one-way seed: inserts version 1 (active) for a slot only if the
    slot has no rows at all. Never overwrites or reactivates anything. A
    per-channel grounding_core slot is seeded from the legacy shared row's active
    content when one exists (so earlier edits carry over), else from the default."""
    legacy = conn.execute(
        "SELECT content FROM prompt_templates WHERE name = %s AND channel = 'shared' AND is_active",
        (GROUNDING_NAME,),
    ).fetchone()
    for (name, channel), content in SEEDS.items():
        if name == GROUNDING_NAME and legacy:
            content = legacy[0]
        conn.execute(
            """
            INSERT INTO prompt_templates (name, channel, version, content, is_active, created_by)
            SELECT %s, %s, 1, %s, true, 'seed'
            WHERE NOT EXISTS (SELECT 1 FROM prompt_templates WHERE name = %s AND channel = %s)
            """,
            (name, channel, content, name, channel),
        )


def _cache_key(channel: str) -> str:
    return f"prompt:{channel}"


def _load_from_db(channel: str) -> tuple[str, dict]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT name, version, content FROM prompt_templates WHERE is_active AND channel = %s "
            "AND name IN (%s, %s)",
            (channel, GROUNDING_NAME, STYLE_NAME),
        ).fetchall()
    by_name = {name: (version, content) for name, version, content in rows}
    if GROUNDING_NAME not in by_name or STYLE_NAME not in by_name:
        raise LookupError(f"no active prompt rows for channel {channel!r} (found {sorted(by_name)})")
    style_version, style = by_name[STYLE_NAME]
    core_version, core = by_name[GROUNDING_NAME]
    return f"{style}\n\n{core}", {GROUNDING_NAME: core_version, STYLE_NAME: style_version}


def get_system_prompt(channel: str = "text") -> tuple[str, dict]:
    """(prompt_text, version_info) for the channel. Never raises."""
    try:
        if channel not in CHANNELS:
            raise ValueError(f"unknown channel {channel!r}")
        cached = get_cached(_cache_key(channel))
        if cached is not MISS and isinstance(cached, dict) and "text" in cached:
            result = (cached["text"], cached["versions"])
        else:
            result = _load_from_db(channel)
            set_cached(_cache_key(channel), {"text": result[0], "versions": result[1]}, ttl=PROMPT_CACHE_TTL_SECONDS)
        _last_good[channel] = result
        return result
    except Exception:
        logger.exception("prompt load failed for channel %r", channel)
    if channel in _last_good:
        logger.error("Using the last successfully loaded %r prompt from process memory", channel)
        return _last_good[channel]
    logger.critical(
        "USING HARDCODED LAST-RESORT PROMPT for channel %r: no prompt could be loaded from "
        "Redis or Postgres and none is held in memory. Check prompt_templates.", channel,
    )
    return LAST_RESORT_PROMPT, {"last_resort": 0}


def refresh_prompt_cache() -> None:
    """Write-through after an admin change: reload both channels from the DB and
    overwrite their cache entries so the next turn uses the new prompt. A shared
    row affects both channels, so both are always refreshed."""
    for channel in CHANNELS:
        try:
            result = _load_from_db(channel)
        except Exception:
            logger.exception("prompt refresh failed for channel %r", channel)
            continue
        _last_good[channel] = result
        set_cached(_cache_key(channel), {"text": result[0], "versions": result[1]}, ttl=PROMPT_CACHE_TTL_SECONDS)
