"""
fast_rp.py
==========
WHAT THIS FILE DOES:
Computes FastRP (Fast Random Projection) graph embeddings over the whole
knowledge graph and writes them onto nodes as `fastrp_embedding`, using
Neo4j's Graph Data Science (GDS) library.

Why this exists: embeddings.py's vectors capture MEANING (what a chunk's
text is about) and BM25 captures KEYWORDS. Neither one captures GRAPH
STRUCTURE — that a chunk is connected to the same Metric/Entity nodes as
several other pieces of strong evidence for a question, even if its own
wording doesn't closely match the question text. FastRP encodes each
node's position in the graph (who it's connected to, and who THEY'RE
connected to) into a dense vector, cheaply, without training a model.

retrieval.py reads `fastrp_embedding` the same way it reads `embedding` —
as one more signal blended into the hybrid score (see config.FASTRP_WEIGHT).

PREREQUISITES:
  - Neo4j Graph Data Science (GDS) plugin installed on your Neo4j server.
    (Neo4j Desktop: Manage -> Plugins -> Graph Data Science Library.
     Docker: add NEO4J_PLUGINS=["graph-data-science"] to your env.)
  - `pip install graphdatascience`
  - Run this AFTER graph_builder.py has finished building the graph
    (including build_temporal_links(), so NEXT_VALUE/PREVIOUS_VALUE/
    YOY_CHANGE/UPDATED_TO/PERSISTED_TO edges exist for FastRP to use —
    it will still run without them, just with less signal).

Run directly:
    python fast_rp.py

Safe to re-run: it drops and recreates its own named GDS in-memory graph
projection (config.FASTRP_GRAPH_NAME) each time, and `SET` on write
overwrites any previous fastrp_embedding rather than accumulating.
"""

import logging
import time

from neo4j import GraphDatabase

import config

logging.basicConfig(level=config.LOG_LEVEL, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from graphdatascience import GraphDataScience

    _GDS_LIB_AVAILABLE = True
except ImportError:
    GraphDataScience = None
    _GDS_LIB_AVAILABLE = False


def _drop_existing_projection(gds, graph_name: str):
    try:
        if gds.graph.exists(graph_name)["exists"]:
            gds.graph.drop(graph_name)
            logger.info("Dropped existing GDS graph projection '%s'.", graph_name)
    except Exception:
        # exists() itself can raise on some GDS versions if the catalog is
        # empty — treat any failure here as "nothing to drop" rather than
        # aborting the whole run over a projection that may not exist.
        logger.debug("No existing projection to drop (or check failed).", exc_info=True)


def _project_graph(gds):
    """
    Projects a native GDS in-memory graph over exactly the node labels /
    relationship types graph_builder.py (and temporal_utils.py) actually
    write — see config.FASTRP_NODE_LABELS / config.FASTRP_REL_TYPES.
    Relationships are projected UNDIRECTED so structural similarity
    propagates both ways (e.g. a Chunk and the Metric it CONTAINS_METRIC
    should influence each other's embedding symmetrically) — FastRP is a
    structural/proximity embedding, not a directed-flow one.
    """
    node_projection = {label: {} for label in config.FASTRP_NODE_LABELS}
    rel_projection = {
        rel: {"orientation": "UNDIRECTED"} for rel in config.FASTRP_REL_TYPES
    }

    graph, result = gds.graph.project(
        config.FASTRP_GRAPH_NAME,
        node_projection,
        rel_projection,
    )

    logger.info(
        "Projected graph '%s': %d nodes, %d relationships (%.1fs).",
        config.FASTRP_GRAPH_NAME,
        result["nodeCount"],
        result["relationshipCount"],
        result["projectMillis"] / 1000,
    )

    if result["nodeCount"] == 0:
        logger.warning(
            "Projected graph has 0 nodes — has graph_builder.py been run yet? "
            "Check config.FASTRP_NODE_LABELS matches your actual schema."
        )

    return graph


def run_fastrp():
    if not _GDS_LIB_AVAILABLE:
        raise RuntimeError(
            "graphdatascience is not installed. Run: "
            "pip install graphdatascience"
        )

    start = time.time()

    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )
    gds = GraphDataScience(driver, database=config.NEO4J_DATABASE)

    try:
        _drop_existing_projection(gds, config.FASTRP_GRAPH_NAME)
        graph = _project_graph(gds)

        if graph.node_count() == 0:
            logger.error("Nothing to embed — aborting before calling fastRP.write.")
            return

        logger.info(
            "Running FastRP: dimensions=%d, iterationWeights=%s",
            config.FASTRP_DIMENSIONS,
            config.FASTRP_ITERATION_WEIGHTS,
        )

        result = gds.fastRP.write(
            graph,
            embeddingDimension=config.FASTRP_DIMENSIONS,
            iterationWeights=config.FASTRP_ITERATION_WEIGHTS,
            randomSeed=config.FASTRP_RANDOM_SEED,
            writeProperty="fastrp_embedding",
        )

        logger.info(
            "FastRP done — wrote %d node embeddings in %.1fs (compute %.1fs).",
            result["nodePropertiesWritten"],
            time.time() - start,
            result["computeMillis"] / 1000,
        )

    finally:
        try:
            if gds.graph.exists(config.FASTRP_GRAPH_NAME)["exists"]:
                gds.graph.drop(config.FASTRP_GRAPH_NAME)
        except Exception:
            logger.debug("Cleanup of GDS projection failed (non-fatal).", exc_info=True)
        driver.close()


if __name__ == "__main__":
    run_fastrp()
