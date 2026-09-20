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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
