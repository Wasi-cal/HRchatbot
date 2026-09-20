"""BGE-M3 embedding generation (dense + sparse) and embedding-model
bookkeeping in Postgres.

Runtime: FlagEmbedding's BGEM3FlagModel (BAAI's own library for this
model), because it exposes dense and sparse vectors from BGE-M3 in a
single .encode() call via return_dense=True, return_sparse=True. That's
cleaner than reconstructing sparse (lexical) weights on top of a plain
sentence-transformers dense-only encoder.

Prefixing: BGE-M3's model card is explicit that, unlike earlier BGE
v1/v1.5 models, it does NOT require an instruction prefix for queries,
and passages/documents never get one either way. Chunk text is embedded
completely unmodified here - no prefix of any kind. (A query-time
instruction prefix, if ever added, belongs in a future retrieval step,
not here.)
"""
import os

import numpy as np
from psycopg.types.json import Jsonb

_model = None


def _load_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        _model = BGEM3FlagModel(model_name, use_fp16=False)
    return _model


class BGEM3Embedder:
    """Thin wrapper: text in, (dense_vecs, sparse_dicts, token_counts) out."""

    def __init__(self):
        self.model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        self.model = _load_model()
        # Determine dimension at runtime from the model itself, rather than
        # hardcoding, so this stays correct if the configured model changes.
        probe = self.model.encode(["_"], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self.dimension = int(probe["dense_vecs"].shape[1])

    def embed(self, texts: list[str], batch_size: int = 12, max_length: int = 8192):
        """Returns (dense: np.ndarray[n, dim], sparse: list[dict[str, float]],
        token_counts: list[int])."""
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"])
        sparse = [
            {str(k): float(v) for k, v in weights.items()}
            for weights in out["lexical_weights"]
        ]
        token_counts = [len(ids) for ids in self.model.tokenizer(texts, truncation=False)["input_ids"]]
        return dense, sparse, token_counts


def ensure_embedding_model_registered(conn, embedder: BGEM3Embedder) -> int:
    """Idempotently registers the embedding model in embedding_models and
    marks it as the sole active row in provider_configs for
    category='embedding'. Returns the embedding_models.id to use for
    chunk_embeddings inserts.
    """
    version = os.environ.get("EMBEDDING_MODEL_VERSION", "v1")

    row = conn.execute(
        "SELECT id FROM embedding_models WHERE name = %s AND version = %s",
        (embedder.model_name, version),
    ).fetchone()
    if row:
        model_id = row[0]
    else:
        row = conn.execute(
            """
            INSERT INTO embedding_models (name, version, dimension, provider, config)
            VALUES (%s, %s, %s, 'local', %s)
            RETURNING id
            """,
            (embedder.model_name, version, embedder.dimension, Jsonb({"runtime": "FlagEmbedding.BGEM3FlagModel"})),
        ).fetchone()
        model_id = row[0]

    # Deactivate any previously active embedding provider first, in the
    # same transaction, so the partial unique index on
    # provider_configs(category) WHERE is_active is never violated.
    conn.execute(
        "UPDATE provider_configs SET is_active = false, updated_at = now() "
        "WHERE category = 'embedding' AND is_active"
    )
    existing = conn.execute(
        "SELECT id FROM provider_configs WHERE category = 'embedding' AND provider_name = %s",
        (embedder.model_name,),
    ).fetchone()
    config = Jsonb({"model_id": model_id, "version": version, "dimension": embedder.dimension})
    if existing:
        conn.execute(
            "UPDATE provider_configs SET is_active = true, config = %s, updated_at = now() WHERE id = %s",
            (config, existing[0]),
        )
    else:
        conn.execute(
            """
            INSERT INTO provider_configs (category, provider_name, config, is_active)
            VALUES ('embedding', %s, %s, true)
            """,
            (embedder.model_name, config),
        )
    return model_id
