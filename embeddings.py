"""
embeddings.py
=============
WHAT THIS FILE DOES:
Gives every embeddable node in Neo4j (Chunk, Table, AudioChunk — anything
with an embedding_text property) a vector embedding, using the
all-MiniLM-L6-v2 model.

This is what lets us later search "what's relevant to this question" by
MEANING instead of exact keyword matching — e.g. a question about
"profitability" can still match a chunk about "net margins" even though
the words are different.

Run directly after graph_builder.py:
    python embeddings.py
"""

import logging
import time

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

import config

logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = config.EMBED_BATCH_SIZE
UPDATE_BATCH_SIZE = 500


def _fetch_pending_nodes(session):
    """
    Every node with an embedding_text and no embedding yet, regardless of
    label (Chunk, Table, AudioChunk, ...). Using elementId avoids relying
    on a chunk_id property existing on every label.
    """

    result = session.run(
        """
        MATCH (n)
        WHERE n.embedding_text IS NOT NULL AND n.embedding IS NULL
        RETURN elementId(n) AS eid, labels(n) AS labels, n.embedding_text AS text
        """
    )

    return [(r["eid"], r["labels"], r["text"]) for r in result]


def _write_embeddings(session, rows):
    """
    rows: list of (element_id, vector). Written in one UNWIND batch per
    UPDATE_BATCH_SIZE rows instead of one query per node.
    """

    for i in range(0, len(rows), UPDATE_BATCH_SIZE):
        batch = rows[i : i + UPDATE_BATCH_SIZE]

        session.run(
            """
            UNWIND $rows AS row
            MATCH (n) WHERE elementId(n) = row.eid
            SET n.embedding = row.vector,
                n.embedding_dim = row.dim
            """,
            rows=[
                {"eid": eid, "vector": vec, "dim": len(vec)} for eid, vec in batch
            ],
        )


def add_embeddings():
    start = time.time()

    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

    try:
        with driver.session(database=config.NEO4J_DATABASE) as session:
            pending = _fetch_pending_nodes(session)

        if not pending:
            logger.info("Nothing to embed — every node is already up to date.")
            return

        by_label = {}
        for _, labels, _ in pending:
            label = labels[0] if labels else "Unknown"
            by_label[label] = by_label.get(label, 0) + 1

        logger.info("Found %d nodes to embed: %s", len(pending), by_label)

        texts = [text for _, _, text in pending]

        try:
            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception:
            logger.error("Embedding model failed on the batch", exc_info=True)
            raise

        rows = [(eid, vec.tolist()) for (eid, _, _), vec in zip(pending, vectors)]

        with driver.session(database=config.NEO4J_DATABASE) as session:
            try:
                _write_embeddings(session, rows)
            except Exception:
                logger.error("Failed writing embeddings to Neo4j", exc_info=True)
                raise

        elapsed = time.time() - start
        logger.info(
            "Done — %d nodes embedded across labels %s in %.1fs.",
            len(rows),
            by_label,
            elapsed,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    add_embeddings()
