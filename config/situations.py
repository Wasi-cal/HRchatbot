"""Situations and options: HR-advisor mode for conversations about a problem (leave shortfall, work from home refused, unwell,
bereavement, stress, a dispute with a manager).

A situation = trigger phrases + levers. A lever is a retrieval query (written from the corpus's own wording) that finds the
policy that offers one way forward (work from home, comp-off, leave without pay, ...). Everything lives in three tables
(situations, situation_triggers, situation_levers) edited through /api/admin/situations; the detector and the lever retrieval
are deterministic (no LLM call) and run only when app_config situations.enabled is true.

Reads on a turn go through the in-process copy (SITUATIONS_TTL seconds) over a Redis cache-aside over the DB, so a turn costs
no extra Redis/DB call except once per TTL. Writes in the admin API call invalidate_situations().
"""
import time

from psycopg.types.json import (
    Jsonb,  # noqa: F401  (kept for symmetry with other config modules)
)

from cache.redis_cache import MISS, get_cached, set_cached
from vectorstore.db import get_pool

SITUATIONS_CACHE_KEY = "situations:v1"
SITUATIONS_TTL = 30
_MEMO: dict = {"expires": 0.0, "value": None}

# name -> (triggers, levers [(name, query_text)]). Lever wordings come from the Employee Handbook's own language (probed against the
# corpus, see eval/situations_test.py); triggers are natural speech and are matched whole-phrase after normalisation
# (lowercase, apostrophes dropped: "I'm" -> "im").
SEED = {
    "leave_shortfall": (
        ["leaves left", "leave left", "leave balance", "out of leave", "run out of leave", "running out of leave", "not enough leave", "no leaves left",
         "only three days", "only 3 days", "only two days", "days of earned leave", "earned leave left", "dont have enough leave"],
        [("earned leave balance and carry forward", "earned leave entitlement carry forward lapse encashment balance"),
         ("work from home approval", "work from home prior approval from manager emergency situations important personal necessities"),
         ("compensatory off", "compensatory leave comp off earn and utilize"),
         ("leave without pay", "leave without pay LWP loss of pay when leave balance is exhausted"),
         ("unplanned absence", "unplanned absence without information or prior permission or sanction")],
    ),
    "wfh_blocked": (
        ["not approving", "wont approve", "wouldnt approve", "refused", "manager rejected", "manager refused", "manager denied", "rejected my",
         "rejected work from home", "refused work from home", "denied work from home", "wfh rejected", "wfh denied", "manager said no", "cant work from home"],
        [("work from home policy", "work from home prior approval from the concerned manager approving authority"),
         ("reasons for work from home", "reasons that could demand work from home bad weather medical reasons health grounds immediate family member short-term medical attention"),
         ("sick leave", "sick leave entitlement 7 days"),
         ("earned leave", "earned leave entitlement 21 days carry forward"),
         ("flexible working", "flexible working timings report to work")],
    ),
    "unwell": (
        ["very sick", "feeling sick", "im sick", "i am sick", "im unwell", "i am unwell", "feeling unwell", "not feeling well", "i have a fever", "cant travel"],
        [("sick leave", "sick leave entitlement 7 days"),
         ("work from home on health grounds", "work from home medical reasons health grounds not fully mobile but fully productive from home"),
         ("earned leave", "earned leave entitlement carry forward"),
         ("sick leave proof in notice period", "during notice period employees can't use leave balance other than sick leave with proof"),
         ("leave without pay", "leave without pay loss of pay when no leave balance")],
    ),
    "bereavement": (
        ["passed away", "my dog died", "my pet died", "my cat died", "died", "funeral", "death in the family", "lost my father", "lost my mother", "lost my grandmother", "bereavement"],
        [("bereavement leave", "bereavement leave 2 days entitlement type of leave"),
         ("earned leave", "earned leave entitlement carry forward"),
         ("work from home for emergencies", "work from home emergency situations important personal necessities not able to report to the office"),
         ("leave without pay", "leave without pay loss of pay when no leave balance")],
    ),
    "stress_wellbeing": (
        ["at my limit", "mental health", "burnt out", "burned out", "burnout", "overwhelmed", "stressed out", "im stressed", "cant cope", "too much pressure", "anxious", "depressed"],
        [("employee well-being", "safety health and well-being of all employees"),
         ("earned leave", "earned leave entitlement carry forward"),
         ("sick leave", "sick leave entitlement 7 days"),
         ("work from home", "work from home arrangement prior approval manager"),
         ("flexible working", "flexible working timings report to work")],
    ),
    "manager_dispute": (
        ["manager rejected", "manager refused", "manager denied", "manager wont", "my manager said no", "manager rejects", "manager is not approving", "not approving"],
        [("work from home approval", "work from home prior approval from the concerned manager approving authority"),
         ("comp off approval", "comp off approval subject to operational requirements and managerial discretion"),
         ("unplanned absence", "unplanned absence without information or prior permission or sanction")],
    ),
}

DEFAULT_APP_CONFIG = {
    "situations.enabled": (False, "Master switch for HR-advisor mode (situation detection, lever retrieval, advisor prompt). Off = every turn behaves exactly as before."),
    "situations.window_messages": (6, "How many of the most recent user messages (plus the current one) are scanned for situation trigger phrases."),
    "situations.max_active": (2, "At most this many situations are active at once, most recently triggered first."),
    "situations.max_levers": (8, "Upper bound on lever queries searched on one turn (levers of the active situations, taken round-robin by position)."),
    "situations.lever_weight": (1.0, "RRF weight of each lever query's ranked lists relative to the main query's."),
    "situations.lever_candidate_n": (2, "Candidates pulled per lever query (dense and sparse each)."),
    "situations.max_chunks_per_doc": (4, "At most this many chunks per document in a situation turn's context (the leave, work-from-home and comp-off rules all live in ONE handbook document)."),
    "retrieval.top_k.situation.text": (6, "Chunks retrieved for a text turn while a situation is active."),
    "retrieval.top_k.situation.voice": (5, "Chunks retrieved for a voice turn while a situation is active."),
    "wellbeing.resources_message": ("", "HR-supplied wellbeing / assistance information the advisor may mention for stress or mental-health turns. Empty = nothing is mentioned (never invented)."),
}


def seed_situations(conn) -> None:
    """Idempotent: inserts the seed situations only when the situations table is empty (HR edits are never overwritten)."""
    if conn.execute("SELECT count(*) FROM situations").fetchone()[0]:
        return
    for name, (triggers, levers) in SEED.items():
        sid = conn.execute("INSERT INTO situations (name, enabled) VALUES (%s, true) RETURNING id", (name,)).fetchone()[0]
        for phrase in triggers:
            conn.execute("INSERT INTO situation_triggers (situation_id, phrase) VALUES (%s, %s) ON CONFLICT DO NOTHING", (sid, phrase))
        for position, (lever_name, query_text) in enumerate(levers, start=1):
            conn.execute("INSERT INTO situation_levers (situation_id, position, name, query_text) VALUES (%s, %s, %s, %s)", (sid, position, lever_name, query_text))


def _load_from_db() -> list[dict]:
    with get_pool().connection() as conn:
        sits = conn.execute("SELECT id, name FROM situations WHERE enabled ORDER BY id").fetchall()
        triggers = conn.execute("SELECT situation_id, phrase FROM situation_triggers ORDER BY id").fetchall()
        levers = conn.execute("SELECT situation_id, position, name, query_text FROM situation_levers ORDER BY situation_id, position").fetchall()
    from channels.common import normalize

    out = []
    for sid, name in sits:
        out.append({
            "name": name,
            "triggers": [normalize(p) for s, p in triggers if s == sid and normalize(p)],
            "levers": [{"position": pos, "name": n, "query_text": q} for s, pos, n, q in levers if s == sid],
        })
    return out


def get_situations() -> list[dict]:
    """Enabled situations with normalised triggers and ordered levers. In-process copy -> Redis -> DB."""
    now = time.monotonic()
    if _MEMO["value"] is not None and _MEMO["expires"] > now:
        return _MEMO["value"]
    cached = get_cached(SITUATIONS_CACHE_KEY)
    if cached is not MISS:
        value = cached
    else:
        value = _load_from_db()
        set_cached(SITUATIONS_CACHE_KEY, value, ttl=SITUATIONS_TTL)
    _MEMO.update(value=value, expires=now + SITUATIONS_TTL)
    return value


def invalidate_situations() -> None:
    from cache.redis_cache import delete_cached

    _MEMO.update(value=None, expires=0.0)
    delete_cached(SITUATIONS_CACHE_KEY)


def detect_situations(current: str, history: list[dict], window: int, max_active: int) -> list[dict]:
    """Active situations for this turn, most recently triggered first, at most max_active. The last `window` USER messages of the
    history plus the current message are scanned; a situation stays active while any of its triggers is still inside that window
    (nothing is stored: it is re-derived from the history every turn). Ties (same message) keep the seed order."""
    from channels.common import _contains_phrase, normalize

    messages = [m["content"] for m in history if m["role"] == "user"][-max(0, window):] + [current]
    normalized = [normalize(m) for m in messages]
    found = []
    for order, situation in enumerate(get_situations()):
        latest = max((i for i, n in enumerate(normalized) if _contains_phrase(n, situation["triggers"])), default=None)
        if latest is not None:
            found.append((-latest, order, situation))
    found.sort(key=lambda x: (x[0], x[1]))
    return [s for _, _, s in found[:max(0, max_active)]]


def lever_queries(active: list[dict], max_levers: int) -> list[str]:
    """Lever query texts of the active situations, round-robin by position, de-duplicated, at most max_levers."""
    out: list[str] = []
    depth = max((len(s["levers"]) for s in active), default=0)
    for i in range(depth):
        for s in active:
            if i < len(s["levers"]):
                q = s["levers"][i]["query_text"]
                if q not in out:
                    out.append(q)
    return out[:max(0, max_levers)]
