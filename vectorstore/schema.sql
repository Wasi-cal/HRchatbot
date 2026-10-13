-- HR RAG chatbot vector store schema.
-- Idempotent: safe to run against an already-provisioned database.
--
-- Versioning model: a document is never mutated or deleted in place.
-- When a source file is re-ingested with different content, the loader
-- marks the previous active document's chunks is_superseded = true and
-- inserts the new document + chunks as fresh rows. This keeps history
-- inspectable and lets chunk_embeddings stay append-only.

CREATE EXTENSION IF NOT EXISTS vector;

-- BGE-M3 dense output is 1024-dimensional. If a different-dimensioned
-- embedding model is introduced later, this column's dimension must
-- change too (pgvector requires a fixed dimension per column) - see
-- README note in vectorstore/embedder.py.
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_file     TEXT NOT NULL,
    document_title  TEXT,
    effective_date  TEXT,   -- free-text date from the source doc (e.g. "07-Nov-2025"); not all source dates are unambiguous, so this is kept as text rather than a strict DATE column
    version         TEXT,   -- free-text version label from the source doc (e.g. "1.0", "2.0"), distinct from effective_date
    status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    is_restricted   BOOLEAN NOT NULL DEFAULT false,  -- manual-only flag (see access_control/) - never set true automatically; retrieve() excludes restricted documents unconditionally
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_restricted BOOLEAN NOT NULL DEFAULT false;

-- Only one active document per source_file at a time - re-ingesting a
-- changed file supersedes the old row rather than replacing it in place.
CREATE UNIQUE INDEX IF NOT EXISTS ux_documents_source_file_active
    ON documents (source_file)
    WHERE status = 'active';

CREATE TABLE IF NOT EXISTS applicability_tags (
    id          SERIAL PRIMARY KEY,
    tag_type    TEXT NOT NULL,   -- e.g. "country", "employment_type"
    tag_value   TEXT NOT NULL,   -- e.g. "US", "contractor"
    UNIQUE (tag_type, tag_value)
);

CREATE TABLE IF NOT EXISTS document_applicability (
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES applicability_tags(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, tag_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id        TEXT NOT NULL,  -- the string id from ingestion JSON, e.g. "All_IT_Policies_0025" - stable within a document version, NOT globally unique (a re-ingested version of the same file reuses the same chunk_id strings)
    section_path    TEXT,
    chunk_type      TEXT NOT NULL CHECK (chunk_type IN ('prose', 'table', 'qa_pair')),
    source_type     TEXT NOT NULL CHECK (source_type IN ('digital_text', 'ocr')),
    ocr_confidence  REAL,
    page_number     INTEGER,
    content_hash    TEXT NOT NULL,
    text            TEXT NOT NULL,
    token_count     INTEGER,
    is_superseded   BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_is_superseded ON chunks (is_superseded);

CREATE TABLE IF NOT EXISTS embedding_models (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,      -- e.g. "BAAI/bge-m3"
    version     TEXT NOT NULL,      -- free-form label, bump when tokenization/settings change for the same model id
    dimension   INTEGER NOT NULL,   -- dense vector width, determined at runtime from the model itself (see embedder.py)
    provider    TEXT NOT NULL CHECK (provider IN ('local', 'api')),
    config      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id                  BIGSERIAL PRIMARY KEY,
    chunk_id            BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding_model_id  INTEGER NOT NULL REFERENCES embedding_models(id) ON DELETE CASCADE,
    dense_vector        vector(1024) NOT NULL,
    sparse_vector       JSONB,  -- token_id (string) -> weight (float) pairs, as returned by BGE-M3's lexical_weights output
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (chunk_id, embedding_model_id)
);

-- HNSW (pgvector >= 0.5.0) with cosine distance ops, matching the cosine
-- similarity search used by vectorstore/verify.py.
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_dense_hnsw
    ON chunk_embeddings USING hnsw (dense_vector vector_cosine_ops);

-- SECRETS NEVER LIVE HERE: API keys (OPENAI_API_KEY, DEEPGRAM_API_KEY),
-- THINK_ENDPOINT_SECRET and DB credentials are env-only, on purpose. This
-- table holds non-secret, per-provider tunables (model names, thresholds,
-- voice, greeting) and is readable via the admin API - do not "centralize"
-- secrets into it.
CREATE TABLE IF NOT EXISTS provider_configs (
    id              SERIAL PRIMARY KEY,
    category        TEXT NOT NULL CHECK (category IN ('embedding', 'generation', 'stt', 'tts')),
    provider_name   TEXT NOT NULL,
    config          JSONB,
    is_active       BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_provider_configs_category_active
    ON provider_configs (category, is_active);

-- Enforce at most one active provider per category at the DB level
-- (belt-and-suspenders alongside the application-level upsert logic in
-- vectorstore/embedder.py, which explicitly deactivates the previous
-- active row before activating a new one in the same transaction).
CREATE UNIQUE INDEX IF NOT EXISTS ux_provider_configs_active_category
    ON provider_configs (category)
    WHERE is_active;

-- Monitoring log for the serving layer (api/). One row per /api/chat
-- call. Written fire-and-forget by api/query_log.py; failures to write
-- never affect the response.
CREATE TABLE IF NOT EXISTS query_logs (
    id                          BIGSERIAL PRIMARY KEY,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    query_text                  TEXT NOT NULL,
    response_type               TEXT NOT NULL CHECK (response_type IN ('answer', 'clarification', 'refusal_shortcut', 'smalltalk', 'source_lookup')),
    answer_text                 TEXT,
    user_attributes             JSONB,    -- legacy: no longer written (applicability layer unwired)
    blocking_tag_type           TEXT,     -- legacy: only on historical response_type = 'clarification' rows
    shortcut_fired              BOOLEAN,  -- NULL for clarification (no generation attempted)
    retrieval_time_ms           DOUBLE PRECISION,
    ttft_ms                     DOUBLE PRECISION,
    total_generation_time_ms    DOUBLE PRECISION,
    chunk_ids_used              JSONB     -- array of chunks.chunk_id strings
);

CREATE INDEX IF NOT EXISTS idx_query_logs_created_at ON query_logs (created_at DESC);

-- Request origin: the text API (api/chat.py) vs the voice think endpoint (api/voice.py).
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'text';
DO $$ BEGIN
    ALTER TABLE query_logs ADD CONSTRAINT query_logs_source_check CHECK (source IN ('text', 'voice'));
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 'smalltalk' (greetings etc. answered without retrieval/LLM) and 'source_lookup' ("where is this from": a
-- templated reply from a deterministic re-retrieval, no LLM; see channels/common.py)
-- was added after the table first shipped; widen the constraint on existing databases.
ALTER TABLE query_logs DROP CONSTRAINT IF EXISTS query_logs_response_type_check;
ALTER TABLE query_logs ADD CONSTRAINT query_logs_response_type_check
    CHECK (response_type IN ('answer', 'clarification', 'refusal_shortcut', 'smalltalk', 'source_lookup'));

-- Pipeline-level tunables, editable at runtime via /api/admin/config
-- (read through a Redis cache, see config/runtime.py). Seeded by
-- config/runtime.py's seed_runtime_config() with the values previously
-- hardcoded in vectorstore/retrieve.py.
-- SECRETS NEVER LIVE HERE: API keys, THINK_ENDPOINT_SECRET and DB
-- credentials are env-only, on purpose. This table is readable via the
-- admin API - do not "centralize" secrets into it.
CREATE TABLE IF NOT EXISTS app_config (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    description TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Generation system prompts, composed per channel from that channel's two active
-- rows: style/<text|voice> + grounding_core/<text|voice> (see config/prompts.py).
-- The 'shared' channel value is legacy (an earlier shared grounding_core row) and unused.
-- Rows are immutable: an edit inserts a NEW version and activates it (see
-- /api/admin/prompts), so history is kept and rollback is just re-activating.
-- Seeded by config/prompts.py via seed_runtime_config(), only for slots with no
-- rows, never overwriting edits.
-- SECRETS NEVER LIVE HERE: prompts must never contain API keys, credentials or
-- other secrets. This table is readable via the admin API and its contents are
-- sent to the LLM provider on every generation call.
CREATE TABLE IF NOT EXISTS prompt_templates (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    channel     TEXT NOT NULL CHECK (channel IN ('shared', 'text', 'voice')),
    version     INTEGER NOT NULL CHECK (version >= 1),
    content     TEXT NOT NULL CHECK (length(btrim(content)) > 0),
    is_active   BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by  TEXT,  -- free-form note: 'seed', or who/why for an admin edit
    CONSTRAINT ux_prompt_templates_version UNIQUE (name, channel, version)
);

-- At most one active row per (name, channel).
CREATE UNIQUE INDEX IF NOT EXISTS ux_prompt_templates_active
    ON prompt_templates (name, channel)
    WHERE is_active;

-- Which prompt rows produced a logged generation, e.g. {"grounding_core": 1, "style": 2}.
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS prompt_version JSONB;

-- Conversation-aware turns: whether prior user messages were folded into the retrieval query
-- (a follow-up) and whether the turn was treated as advisory (broader, multi-policy context).
-- NULL on rows logged before these columns existed.
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS contextualized BOOLEAN;
ALTER TABLE query_logs ADD COLUMN IF NOT EXISTS advisory BOOLEAN;
