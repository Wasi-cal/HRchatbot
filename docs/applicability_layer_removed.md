# Applicability filtering / clarification layer - unwired

**Last commit that still contains the working layer: `fd317af`**
(`Add small talk, per-channel text/voice pipelines, DB-stored prompts, conversation-aware advisory retrieval`).
Everything below can be read with `git show fd317af:<path>`.

Nothing in the database was deleted. `applicability_tags`, `document_applicability`, the per-document tags,
`documents.is_restricted` and the `query_logs` columns are all still there.

## What the layer did

1. Each uploaded document was tagged (country, employment type) by an LLM at upload time.
2. Every chat/voice turn ran an **unfiltered probe retrieval**, looked at the applicability tags of the retrieved
   documents and, if the user's answer would change which documents apply and the user's attributes were unknown,
   replied with a **clarifying question** ("Are you a full-time employee or a contractor?") instead of an answer.
3. The user's free-text reply was turned into an attribute value by an **LLM call** (`resolve_clarification`), after
   which the real, **filtered** retrieval ran (documents whose tags excluded the user were dropped before scoring).
4. Text: the client kept `user_attributes` and `pending_clarification`. Voice: the endpoint re-derived all of it from the
   message history (fixed lead-in "Quick check before I answer - ", closed question wordings, a Redis cache of
   classified Q/A pairs, `country=India` default).

## What is unwired now

A turn is: small-talk match -> **retrieve once** -> **generate once** -> stream. `retrieve()` still excludes
`is_superseded` chunks and `is_restricted` documents, and nothing else. Everyone is answered from everything that is not
restricted.

| Where (at `fd317af`) | What held the logic | Now |
|---|---|---|
| `vectorstore/retrieve.py` | `_applicability_filter_sql()` and the `user_attributes` parameter of `dense_search`, `sparse_search`, `sparse_search_many`, `retrieve` | replaced by `_visibility_filter_sql()` (restricted only); `retrieve(user_attributes=...)` is accepted and ignored |
| `channels/common.py` | `with_default_attributes()`, `detect_blocking_tag_types()` | removed |
| `channels/text.py`, `channels/voice.py` | `CLARIFYING_QUESTIONS`, the probe + `needs_clarification` branch in `answer_stream()`, `default_attributes()`, the `needs_clarification` return of `answer()` | `answer_stream()` is now the single retrieve+generate flow |
| `access_control/ask_and_answer.py` | `CLARIFYING_QUESTIONS`, `detect_blocking_tag_types` re-export | thin router only; `user_attributes` ignored |
| `access_control/resolve_clarification.py` | the clarification-reply LLM call | file deleted |
| `access_control/user_attributes.py` | documented attribute shape, `KNOWN_TAG_TYPES` | file deleted |
| `api/chat.py` | `PendingClarification`, `user_attributes`, `ClarificationResponse`, echoed attributes | request/response shapes are `answer` / `refusal_shortcut` / `smalltalk` only; unknown request fields are ignored |
| `api/voice.py` | `CLARIFY_LEAD_IN`, `QUESTION_VARIANTS`, `parse_clarification`, `_clarif_at`, `_original_query[_index]`, `_resolve_value` (Redis `clarif:` cache), `attributes_from_history`, `DEFAULT` country, clarification-pair skipping in `conversation_history` | removed; `conversation_history()` only drops the synthetic `-` turn and the greeting |
| `api/admin.py` | `_default_attributes` validator and the `user.default_attributes[.text\|.voice]` config keys | removed |
| `config/runtime.py`, `config/channel_settings.py` | seed of `user.default_attributes` and its per-channel copies | removed (existing rows are left in `app_config`, unused) |
| `api/query_log.py` | `user_attributes`, `blocking_tag_type` parameters | no longer written |
| `frontend/` (not in git - see appendix) | `userAttributes` state, `pending_clarification`, `.msg.clarify`, the "Known attributes" line, the clarification-rate card | removed |

### Left in place (unused, safe to ignore or drop later)
* `query_logs.user_attributes`, `query_logs.blocking_tag_type` columns (nullable); `response_type = 'clarification'` stays
  allowed by the CHECK constraint so historical rows remain readable. The usage summary still reports historical
  clarification counts; the admin usage view shows them as "Clarifications (historical)".
* `app_config` rows `user.default_attributes`, `user.default_attributes.text`, `user.default_attributes.voice`.
* Redis keys `hrchatbot:clarif:*` (expire never; safe to `DEL`).

### Kept and still used
`applicability_tags`, `document_applicability`, upload-time tag extraction (`access_control/extract_applicability.py`,
once per upload), `access_control/verify_applicability.py`, `documents.is_restricted` and its admin toggle, and the tags
column of the admin documents table. Tags are also used for **labels, not filtering**: `fetch_chunk_details()` returns a
plain per-document label (e.g. `India; full-time employee, intern`) in the same query that fetches chunk details, and
`channels/common.py build_messages()` prefixes each excerpt with `[Applies to: ...]` so the model can say which group a
rule applies to.

## Re-enabling it (role-based answers)

1. Restore the code from the last commit that had it:
   ```bash
   git show fd317af:access_control/resolve_clarification.py > access_control/resolve_clarification.py
   git show fd317af:access_control/user_attributes.py       > access_control/user_attributes.py
   ```
2. `vectorstore/retrieve.py`: bring back `_applicability_filter_sql(user_attributes)` from `fd317af` (it returns
   `NOT d.is_restricted` plus one NOT-EXISTS clause per known attribute), give `dense_search`, `sparse_search`,
   `sparse_search_many` their `user_attributes` argument again and pass `retrieve()`'s `user_attributes` through.
   Keep the new `applies_to` column in `fetch_chunk_details()` - it is independent.
3. `channels/common.py`: restore `with_default_attributes` and `detect_blocking_tag_types` from `fd317af`.
4. `channels/text.py`, `channels/voice.py`: restore `CLARIFYING_QUESTIONS`, `default_attributes()` and the old
   `answer_stream()` wrapper (probe -> `detect_blocking_tag_types` -> `needs_clarification` event -> generation with
   `user_attributes`). Re-add `user_attributes` to `answer()` and the `needs_clarification` early return. Re-add the
   `user.default_attributes` seed (`config/runtime.py`) and validator (`api/admin.py`), or delete the stale rows' need.
5. `api/chat.py` / `api/voice.py`: restore `PendingClarification`, `ClarificationResponse`, the `resolve_clarification`
   flow and, for voice, the history parsing (`parse_clarification`, `_resolve_value`, `attributes_from_history`).
   `api/query_log.py`: write `user_attributes` and `blocking_tag_type` again.
6. Prompts: create new `prompt_templates` versions that again allow/require the clarification flow, and drop the rule
   "Never ask the user a clarifying question about their group, location or employment type" from `grounding_core`.
   (Rows are immutable; roll back with `POST /api/admin/prompts/{id}/activate` for the versions before the unwiring.)
7. Frontend: restore the snippets in the appendix. 8. Run `eval/run_regression.py` from `fd317af`'s version of the access
   control scenarios (`git show fd317af:eval/run_regression.py`) to confirm the filtering behaves as before.

A faster route for the backend: `git diff fd317af HEAD -- <paths above>` shows exactly what the unwiring changed, and
`git checkout fd317af -- <path>` restores a whole file (then re-apply the unrelated changes made since, such as the
`applies_to` label).

## Appendix - removed frontend code (frontend/ was never committed)

`frontend/index.html`:
```html
<p class="muted small">Known attributes: <code id="attrs">{}</code>
  <button id="reset-btn" type="button" class="link">reset conversation</button></p>
```
`frontend/app.css`:
```css
.msg.clarify { border-left: 4px solid #e0a000; }
```
`frontend/app.js` (text chat):
```js
const state = { userAttributes: {}, pending: null, history: [] };
function renderAttrs() { $("#attrs").textContent = JSON.stringify(state.userAttributes); }
// reset handler also did: state.userAttributes = {}; state.pending = null; renderAttrs();

// on submit
const body = { query, user_attributes: state.userAttributes, history: state.history.slice(-6) };
const askedQuery = state.pending ? state.pending.original_query : query;
if (state.pending) body.pending_clarification = state.pending;
state.pending = null;
// ...after the response
if (res.user_attributes) state.userAttributes = res.user_attributes;   // echoed, incl. just-resolved attributes
renderAttrs();
if (res.type === "needs_clarification") {
  addMessage("assistant", res.question, "clarify");
  state.pending = { original_query: res.original_query, blocking_tag_type: res.blocking_tag_type };
} else if (res.type === "refusal_shortcut") {
  addMessage("assistant", res.text, "refusal");
  state.history.push({ role: "user", content: askedQuery }, { role: "assistant", content: res.text });
} else {
  addMessage("assistant", res.text);
  state.history.push({ role: "user", content: askedQuery }, { role: "assistant", content: res.text });
}
// usage cards
card("Clarifications", s.by_response_type.clarification),
card("Clarification rate", `${s.clarification_rate}%`),
```
