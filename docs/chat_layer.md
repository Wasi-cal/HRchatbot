# Chat layer (text + voice)

## Chat layer acceptance

Tick all of these before calling the chat layer done:

- [ ] `python3 -m eval.run_regression` is green (retrieval baseline byte-identical).
- [ ] New tests are green: `eval.catalog_test`, `eval.documents_test`, `eval.followups_test`, `eval.smalltalk_test`, `eval.prompts_test`,
      `eval.channel_separation_test`, `eval.advisory_test`, the local-only jsdom checks (`eval/ui/ui_checks.js`, not in the repo, needs the local `frontend/`) and, against a running
      server, `eval.config_test`, `eval.voice_think_test`, `eval.api_smoke_test`.
- [ ] `eval.turn_calls` matches `eval/turn_calls_baseline.json` for every turn type (asserted by `catalog_test` and `documents_test`).
- [ ] The quality set was **manually graded**: `python3 -m eval.run_quality_eval` (and `--channel voice`), then log the result in `eval/GRADING_LOG.md`.
      The set contains every real transcript query from this project (leave policy, interns, the hometown trip with 3 leaves left, the brief
      referral program, a friend applying for an unlisted post, "whats the source of this", "yes thank you", "yessir", a bare "yes" after an offer,
      "we need a standardized tech stack", "wassup", "what docs do u have access to rn").

## Per-turn pipeline

Both channels run the same order; each channel is its own module (`channels/text.py` for `/api/chat`, `channels/voice.py` for the Deepgram
think endpoint in `api/voice.py`) and never imports the other. Shared, behaviour-free helpers live in `channels/common.py` and
`channels/catalog.py`.

1. **Small talk** - whole-utterance match against `smalltalk.phrases.<channel>`. A bare affirmation after an assistant question or offer is NOT small
   talk (it continues the conversation). Reply is canned. `response_type = smalltalk`.
2. **Source lookup** ("where is this from") - only if the history has an earlier assistant answer. Re-runs the previous turn's retrieval through the
   same `_retrieve()` and names up to 3 documents from a template. `response_type = source_lookup`.
3. **Catalog** ("what documents do you have") - whole-utterance matcher over `catalog.*`; reply from the cached title list. `response_type = catalog`.
4. **Retrieval shortcut** - the turn is planned (history cleaned, follow-up query built, advisory detected), retrieved once, and if the top fused score is
   below `generation.confidence_threshold.<channel>` the canned "I couldn't find anything on that in the HR policies I have. I can help with things
   like ..." reply is returned. `response_type = refusal_shortcut`.
5. **Retrieval + generation** - one retrieval, one streamed generation call with the channel's prompt (style + grounding_core) and the cleaned history.
   `response_type = answer`.

There is no applicability filtering and no clarifying question (see "Intentionally removed").

### Calls per turn type (steady state, measured by `eval/turn_calls.py`)

| turn | LLM | retrieval | embedding | DB | Redis |
|---|---|---|---|---|---|
| small talk, thanks / "yes thank you" after a statement | 0 | 0 | 0 | 1 (log insert) | 0 |
| small talk "what can you do" (cached topic list) | 0 | 0 | 0 | 1 | 1 |
| catalog | 0 | 0 | 0 | 1 | 0 |
| source lookup | 0 | 1 | 1 | 5 | 6 |
| retrieval shortcut | 0 | 1 | 1 | 5 | 6 |
| normal answer (text, voice) | 1 | 1 | 1 | 5 | 7 |
| advisory follow-up (batched expansion embedding) | 1 | 1 | 1 | 8 | 17 |
| bare "yes" after an offer | 1 | 1 | 1 | 5 | 11 |

DB / Redis counts are the existing config-cache reads and the response-log insert; none of this task's features add a call to any turn type. The
keys read on every turn go through a 30-second in-process copy (`config.runtime.memo_config`), so a Redis read happens once per key per 30s, not per
turn. Label norms, catalog titles and topics use the same in-process copy over Redis over the DB.

## app_config keys (chat layer)

Seeded once by `config.runtime.seed_runtime_config()` (never overwriting an edit); the DB is the source of truth afterwards. Every key is editable
with `PATCH /api/admin/config/{key}` and validated server-side; reads are cache-aside (Redis, 30s) with the in-process copy described above.

### Retrieval (global; each also has a `.text` / `.voice` copy seeded from it, see channel settings)

| key | seed | meaning |
|---|---|---|
| `retrieval.top_k` | `3` | Chunks returned by retrieve() after RRF fusion. |
| `retrieval.candidate_n` | `20` | Candidates pulled from each of dense and sparse search before fusion. |
| `retrieval.rrf_k` | `60` | Reciprocal Rank Fusion constant k. |
| `retrieval.top_k.advisory.text` | `6` | Chunks retrieved for an advisory text turn. |
| `retrieval.top_k.advisory.voice` | `4` | Chunks retrieved for an advisory voice turn. |

### Conversation context, follow-ups, advisory turns

| key | seed | meaning |
|---|---|---|
| `conversation.history_messages` | `6` | Max prior user/assistant messages accepted per turn (server-side cap on client/Deepgram history). |
| `conversation.history_max_chars` | `400` | Each history message is truncated to this many characters. |
| `retrieval.context.max_words_followup` | `8` | An utterance shorter than this many words is treated as a follow-up and gets prior user turns prepended to its retrieval query. |
| `retrieval.context.prior_user_turns` | `2` | How many previous user messages are prepended to a follow-up's retrieval query. |
| `retrieval.context.followup_markers` | `["that", "it", "this", "those", "these", "them", "there", "same", "an…` | Words/phrases that mark an utterance as a follow-up (whole-word match on the normalized text). |
| `retrieval.advisory.markers` | `["anything else i can do", "anything else that i can do", "what are m…` | Phrases (normalized: lowercase, no punctuation) that mark a turn as advisory: broader, multi-policy retrieval. |
| `retrieval.expansions` | `[{"topic": "leave_shortage", "triggers": ["lop", "loss of pay", "leav…` | Topic -> expansion queries for advisory turns. A topic fires when any trigger phrase appears in the current utterance or prior user messages; its queries are embedded in one batched encode and fused in. |
| `retrieval.expansion.enabled` | `true` | Whether advisory turns run the expansion queries (per channel copies override). |
| `retrieval.expansion.weight` | `2.0` | RRF weight of each expansion query's ranked lists relative to the main query's. |
| `retrieval.expansion.candidate_n` | `2` | Candidates pulled per expansion query (dense and sparse each): only its top hits should steer the context. |
| `retrieval.advisory.max_chunks_per_doc` | `4` | Per-document cap on chunks in an advisory turn's context, applied after RRF fusion. |

### Small talk

| key | seed | meaning |
|---|---|---|
| `smalltalk.phrases.text` | `{"greeting": ["hi", "hii", "hello", "helloo", "hey", "heyy", "heya", …` | Text channel small-talk phrases by category (normalized; whole-utterance match). Editable. |
| `smalltalk.phrases.voice` | `{"greeting": ["hi", "hii", "hello", "helloo", "hey", "heyy", "heya", …` | Voice channel small-talk phrases by category (normalized; whole-utterance match). Editable. |
| `smalltalk.affirmations` | `["yes", "yeah", "yep", "yup", "yea", "sure", "ok", "okay", "yessir", …` | Affirmation phrases: NOT small talk when the previous assistant turn ended with a question or made an offer - they go through the normal pipeline so the offer is followed up. |
| `smalltalk.offer_markers` | `["would you like", "want me to", "i can also"]` | Phrases that mark an assistant message as an offer (a message ending in '?' also counts). |

### Source lookup

| key | seed | meaning |
|---|---|---|
| `source.markers` | `["source", "the source", "whats the source", "what is the source", "w…` | Whole-utterance phrases (normalized, <= 8 words) that ask where the previous answer came from. |
| `source.templates.text` | `{"one": "That came from the {title}.", "one_with_section": "That came…` | Text replies for a source question. Keys: one, one_with_section, many, none, not_found. |
| `source.templates.voice` | `{"one": "That's from the {title} policy.", "many": "That drew on the …` | Voice replies for a source question (document name only). Keys: one, many, none, not_found. |

### Catalog

| key | seed | meaning |
|---|---|---|
| `catalog.enabled` | `true` | Whether 'what documents do you have' questions get the deterministic catalog reply. |
| `catalog.vocab` | `["document", "documents", "policy", "policies", "topic", "topics", "f…` | A catalog question may contain ONLY these content words (after stopword removal and the normalize map). |
| `catalog.nouns` | `["document", "documents", "policy", "policies", "topic", "topics", "f…` | A catalog question must contain at least one of these (multi-word entries match the whole utterance). |
| `catalog.normalize_map` | `{"u": "you", "ur": "your", "docs": "documents", "doc": "document", "r…` | Informal forms expanded before matching; an empty replacement drops the word (u->you, docs->documents, rn->drop). |
| `catalog.stopwords` | `["the", "a", "an", "of", "to", "do", "does", "did", "is", "are", "am"…` | Function words ignored by the catalog vocabulary test (moved here from code; same list). |
| `catalog.max_words` | `12` | A catalog question is at most this many words. |
| `catalog.text_max_titles` | `12` | Titles listed in a text catalog reply before 'and N more'. |
| `catalog.voice_max_titles` | `4` | Short spoken names listed in a voice catalog reply. |
| `catalog.voice_max_title_words` | `5` | Titles longer than this many words are not read aloud. |
| `catalog.templates.text` | `{"main": "I can answer questions from {n} HR documents: {list}.", "mo…` | Text catalog reply parts: main ({n},{list}), more ({k}), topics ({topics}), closing, empty. Must not end with a question or offer. |
| `catalog.templates.voice` | `{"main": "I have {n} HR policy documents, including {names}.", "closi…` | Voice catalog reply parts: main ({n},{names}), closing, empty. Must not end with a question or offer. |

### Group labels

| key | seed | meaning |
|---|---|---|
| `label.mode` | `"deviation"` | Group label on retrieved chunks: deviation (only types where the document differs from the norm) \| always \| off. |
| `label.norm_min_share` | `0.5` | A tag-value set counts as the norm for its type only if at least this share of documents with that type carry it; otherwise the type is always shown. |

### Duplicates and titles

| key | seed | meaning |
|---|---|---|
| `dup.enabled` | `true` | Upload-time duplicate detection (exact fingerprint 409 and possible_duplicates). |
| `dup.jaccard_threshold` | `0.6` | Flag a possible duplicate when the 5-word-shingle Jaccard of two documents is at least this. |
| `dup.containment_threshold` | `0.75` | ...or when either document's shingles are contained in the other's at least this much. |
| `dup.title_match` | `true` | ...or when the normalized titles are equal. |
| `title.generic_phrases` | `["document classification", "confidential", "internal", "table of con…` | A detected title containing one of these phrases triggers a title_warning at upload (never changed automatically). |

### Per-channel copies (seeded from the global value, then independent)

| key | seed |
|---|---|
| `conversation.history_max_chars.text` | `400` |
| `conversation.history_max_chars.voice` | `400` |
| `conversation.history_messages.text` | `6` |
| `conversation.history_messages.voice` | `6` |
| `generation.confidence_threshold.text` | `0.02` |
| `generation.confidence_threshold.voice` | `0.02` |
| `generation.model.text` | `"gpt-4o-mini"` |
| `generation.model.voice` | `"gpt-4o-mini"` |
| `retrieval.advisory.max_chunks_per_doc.text` | `4` |
| `retrieval.advisory.max_chunks_per_doc.voice` | `4` |
| `retrieval.candidate_n.text` | `20` |
| `retrieval.candidate_n.voice` | `20` |
| `retrieval.context.max_words_followup.text` | `8` |
| `retrieval.context.max_words_followup.voice` | `8` |
| `retrieval.context.prior_user_turns.text` | `2` |
| `retrieval.context.prior_user_turns.voice` | `2` |
| `retrieval.expansion.enabled.text` | `true` |
| `retrieval.expansion.enabled.voice` | `true` |
| `retrieval.rrf_k.text` | `60` |
| `retrieval.rrf_k.voice` | `60` |
| `retrieval.top_k.advisory.text` | `6` |
| `retrieval.top_k.advisory.voice` | `4` |
| `retrieval.top_k.text` | `3` |
| `retrieval.top_k.voice` | `3` |
| `user.default_attributes.text` | `{"country": "India"}` |
| `user.default_attributes.voice` | `{"country": "India"}` |

## Prompt rows (`prompt_templates`, immutable versions)

A channel's prompt is its `style` row followed by its `grounding_core` row. Edit with `POST /api/admin/prompts` (new version, activated
atomically), roll back with `POST /api/admin/prompts/{id}/activate`. Never put secrets in a prompt.

| name | channel | active |
|---|---|---|
| `grounding_core` | text | v6 |
| `grounding_core` | voice | v6 |
| `style` | text | v3 |
| `style` | voice | v3 |

History: grounding_core v1 seed; v2 conversation/advisory rules; v3 group labels and no clarifying questions; v4 never infer the user's group, no
unsupported "not entitled", leave-day counting; v5 never refer users to IT/legal/finance/other teams unless the text says so; v6 deviation-only labels
and never address the user as a group they haven't named. Earlier versions are kept for rollback.

## query_logs

`id`, `created_at`, `query_text`, `response_type`, `answer_text`, `user_attributes`, `blocking_tag_type`, `shortcut_fired`, `retrieval_time_ms`, `ttft_ms`, `total_generation_time_ms`, `chunk_ids_used`, `source`, `prompt_version`, `contextualized`, `advisory`

`response_type` is one of `answer`, `refusal_shortcut`, `smalltalk`, `source_lookup`, `catalog` (and the historical `clarification`).
`prompt_version` records the prompt rows used by a generation call; `contextualized` / `advisory` record how retrieval was planned;
`user_attributes` and `blocking_tag_type` are legacy columns that are no longer written. `source` is `text` or `voice`.

## Group labels (deviation only)

Applicability tags are used as LABELS on retrieved chunks, never as filters. Per tag type the norm is the most common tag-value set among active,
unrestricted documents that carry that type (today: country {India}, employment_type {full-time employee}). A document's label shows a type only
when its set differs from the norm, with the full set, values only (`[Applies to: full-time employee, intern]`); no deviation, no label. `label.mode`
(deviation / always / off) and `label.norm_min_share` control it. The tags come with the chunk-detail query and the norms from the cache, so there is
no extra call per turn.

## Document management

- **Duplicates** (upload time and on demand only): `documents.content_fingerprint` is a sha256 over the ordered normalized chunk texts. A different
  source_file with an exact fingerprint match is rejected with 409 (unless `allow_duplicate=true`); otherwise the upload is ingested and
  `possible_duplicates` lists documents flagged by Jaccard >= `dup.jaccard_threshold`, containment (either direction) >= `dup.containment_threshold`,
  or equal normalized titles. Two versions of one policy have a middling Jaccard but a high containment, which is why both are checked
  (the real referral pair: Jaccard 0.68, containment 0.79 / 0.83; every other pair is below 0.19 / 0.32). `GET /api/admin/documents/duplicates` scans
  active documents with a diff summary. Nothing is ever superseded, retitled or retagged automatically.
- **Status**: `PATCH /api/admin/documents/{id}/status` sets `documents.status` and `chunks.is_superseded` together; reversible.
- **Titles**: `PATCH /api/admin/documents/{id}/title` (trimmed, 1-200 chars, one line of plain text; HTML is stored and rendered as text).
  Uploads return `title_warning` for ALL-CAPS titles, titles matching `title.generic_phrases`, or titles equal to a line repeated on 3+ pages.
- **Restrict toggle**: `PATCH /api/admin/documents/{id}/restrict`.

### The shared cache-invalidation helper

`config.doc_caches.invalidate_document_caches()` is the ONE call every upload, status change, title edit, restrict toggle and tag change makes. It
clears the topic list, the catalog title list, the label norms, the cached duplicate scan (admin badges) and this process's in-process copies. Other
workers' in-process copies expire within 30 seconds. Tags are read with each chunk's details, so there is no separate per-document tag cache.

## Intentionally removed

Applicability FILTERING and the clarifying-question flow were unwired (nothing deleted from the database). What was removed, which files held it and how
to re-enable it are in [applicability_layer_removed.md](applicability_layer_removed.md); the last commit that still contained the working layer is
`3f43596`.

## Known limitations

- **Prompt rules are probabilistic.** The model can still occasionally misphrase (for example assign a group to the user or give a different unpaid-day
  count); measured rates are in the quality evals, and the quality set needs manual grading. Rules live in editable prompt rows, not code.
- **Source lookup reports retrieved documents, not the sentences used.** It re-runs the previous turn's retrieval; the model may not have used every
  retrieved document. It can only refer to turns still present in the (capped) history.
- **Duplicates are flagged and resolved by the admin.** Detection is upload-time or on-demand; thresholds are configurable but a heuristic.
- **No per-user history persistence.** History is whatever the client (text) or Deepgram (voice) sends each turn, capped and truncated server-side;
  nothing is stored per user.
- Small-talk, catalog and source matching are deterministic whole-utterance matchers; phrasing outside the configured lists falls through to the
  normal pipeline.
- The admin UI and API have no authentication (dev build).
