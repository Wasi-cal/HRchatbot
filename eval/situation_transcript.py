"""Replays the real situation transcript as a multi-turn conversation on both channels (real model, real retrieval)."""
import json

from dotenv import load_dotenv

load_dotenv('/Users/wasi/InternProjects/HRchatbot/.env')
from channels import get_channel
from vectorstore.db import close_pool, get_pool

TURNS = ["My dog died.", "So if it was a relative, I could have gotten a bereavement leave.", "It's telling me that my pet is not related to me at all.",
         "I'm very sick, can't travel, my manager rejected work from home.", "My birthday is next week, can I take leave that day?",
         "I plan to go to my hometown for a week but I only have three days of earned leave left, is there anything else we can do?",
         "Well, I don't want to take leave without pay.", "And what about the other two days?", "Can I take work from home", "I'm at my limit, it's affecting my mental health."]

def docs(ids):
    with get_pool().connection() as c:
        return {r[0]: r[1] for r in c.execute("SELECT c.chunk_id, d.document_title FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.status='active' AND c.chunk_id=ANY(%s)", (ids,)).fetchall()}

out = {}
for ch in ("text", "voice"):
    m = get_channel(ch); hist = []; rows = []
    for t in TURNS:
        r = m.answer(t, history=hist)
        ids = r.get("chunk_ids", []); d = docs(ids)
        rows.append({"turn": t, "type": r.get("type"), "chunks": [(i[-9:], d.get(i, "?")[:22]) for i in ids], "advisory": r.get("advisory"), "contextualized": r.get("contextualized"),
                     "prompt_version": r.get("prompt_version"), "answer": r["answer"]})
        hist += [{"role": "user", "content": t}, {"role": "assistant", "content": r["answer"]}]
    out[ch] = rows
with open('/Users/wasi/InternProjects/HRchatbot/eval/results/situation_transcript_before.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
for ch, rows in out.items():
    print("=====", ch)
    for i, r in enumerate(rows, 1):
        print(f"\nT{i} {r['turn']!r}\n  adv={r['advisory']} ctx={r['contextualized']} prompt={r['prompt_version']} type={r['type']}\n  chunks={r['chunks']}\n  A: {r['answer']}")
close_pool()
