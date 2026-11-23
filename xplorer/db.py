"""MongoDB wiring: collection setup, lazily-created per-model vector indexes,
a shared text index, and the hybrid (vector + full-text) search used by both the
/search route and the chat agent. Falls back to client-side Reciprocal Rank
Fusion when $rankFusion isn't available on the server.
"""
import logging
import time

from pymongo.errors import OperationFailure

logger = logging.getLogger("xplorer.db")

TEXT_INDEX_NAME = "default_text_index"
_RRF_K = 60

# Fields returned to the UI / agent (never the big embedding vectors).
_PROJECT = {
    "_id": 0,
    "file_path": 1,
    "item_name": 1,
    "llm_analysis": 1,
    "code": 1,
    "tags": 1,
}


def setup_database_and_collection(client, db_name, coll_name):
    """Get the database and create required collections if missing."""
    try:
        db = client[db_name]
        state_coll_name = f"{coll_name}_state"

        if coll_name not in db.list_collection_names():
            logger.warning(f"Collection '{coll_name}' not found. Creating it now...")
            db.create_collection(coll_name)
        collection = db[coll_name]

        if state_coll_name not in db.list_collection_names():
            db.create_collection(state_coll_name)
            db[state_coll_name].create_index("codebase_id")
        state_collection = db[state_coll_name]

        return db, collection, state_collection
    except OperationFailure as e:
        logger.error(f"MongoDB operation failed during setup: {e.details}")
        return None, None, None
    except Exception as e:
        logger.error(f"Unexpected error during database setup: {e}")
        return None, None, None


def _existing_search_indexes(coll):
    try:
        return {idx["name"]: idx for idx in coll.list_search_indexes()}
    except Exception as e:
        logger.warning(f"Could not list search indexes: {e}")
        return {}


def ensure_text_index(coll):
    """Create the shared full-text search index if it doesn't exist."""
    existing = _existing_search_indexes(coll)
    if TEXT_INDEX_NAME in existing:
        return
    try:
        coll.create_search_index(model={
            "name": TEXT_INDEX_NAME,
            "definition": {
                "mappings": {
                    "dynamic": True,
                    "fields": {"tags": {"type": "string", "analyzer": "lucene.keyword"}},
                }
            },
        })
        logger.info(f"Text index '{TEXT_INDEX_NAME}' creation initiated.")
    except OperationFailure as e:
        logger.error(f"Failed to create text index: {e.details}")
    except Exception as e:
        logger.error(f"Unexpected error creating text index: {e}")


def ensure_vector_index(coll, em, dims):
    """Create a vectorSearch index for a specific embedding model if missing.
    Returns the index name."""
    name = em.index_name
    existing = _existing_search_indexes(coll)
    if name in existing:
        return name
    try:
        coll.create_search_index(model={
            "name": name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {"type": "vector", "path": em.field_path,
                     "numDimensions": dims, "similarity": "cosine"},
                    {"type": "filter", "path": "codebase_id"},
                    {"type": "filter", "path": "tags"},
                ]
            },
        })
        logger.info(f"Vector index '{name}' ({dims}d) creation initiated for {em.id}.")
    except OperationFailure as e:
        logger.error(f"Failed to create vector index {name}: {e.details}")
    except Exception as e:
        logger.error(f"Unexpected error creating vector index {name}: {e}")
    return name


def index_is_ready(coll, name):
    for idx in _existing_search_indexes(coll).values():
        if idx.get("name") == name:
            return bool(idx.get("queryable")) or idx.get("status") == "READY"
    return False


def wait_for_index_ready(coll, name, timeout=180, interval=3):
    """Block until a search index is queryable (or timeout)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if index_is_ready(coll, name):
            return True
        time.sleep(interval)
    logger.warning(f"Timed out waiting for index '{name}' to become ready.")
    return False


def index_states(coll):
    """Compact list of search-index states for the health endpoint."""
    out = []
    for idx in _existing_search_indexes(coll).values():
        out.append({
            "name": idx.get("name"),
            "status": idx.get("status"),
            "queryable": bool(idx.get("queryable")),
            "type": idx.get("type", "search"),
        })
    return out


def _vector_stage(em, query_vector, codebase_id, limit, num_candidates=150):
    return {
        "$vectorSearch": {
            "index": em.index_name,
            "path": em.field_path,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
            "filter": {"codebase_id": {"$eq": codebase_id}},
        }
    }


def _text_stage(query_text, codebase_id, limit):
    return [
        {
            "$search": {
                "index": TEXT_INDEX_NAME,
                "compound": {
                    "must": [{
                        "text": {
                            "query": query_text,
                            "path": [
                                "item_name", "code", "tags",
                                "llm_analysis.summary", "llm_analysis.description",
                                "llm_analysis.key_entities",
                            ],
                        }
                    }],
                    "filter": [{"term": {"path": "codebase_id", "query": codebase_id}}],
                },
            }
        },
        {"$limit": limit},
    ]


def hybrid_search(coll, codebase_id, query_text, query_vector, em, limit=10):
    """Hybrid vector + full-text search scoped to one codebase.

    Tries MongoDB's native $rankFusion first; if the server doesn't support it,
    runs the two pipelines separately and fuses them client-side with RRF.
    Returns a list of result docs with a 'score' field.
    """
    try:
        return _rank_fusion_search(coll, codebase_id, query_text, query_vector, em, limit)
    except OperationFailure as e:
        details = str(getattr(e, "details", e))
        logger.info(f"$rankFusion path failed ({details[:160]}); using client-side RRF fallback.")
        return _client_rrf_search(coll, codebase_id, query_text, query_vector, em, limit)


def _rank_fusion_search(coll, codebase_id, query_text, query_vector, em, limit):
    pipeline = [
        {
            "$rankFusion": {
                "input": {
                    "pipelines": {
                        "vectorPipeline": [_vector_stage(em, query_vector, codebase_id, limit)],
                        "fullTextPipeline": _text_stage(query_text, codebase_id, limit),
                    }
                },
                "combination": {"weights": {"vectorPipeline": 0.7, "fullTextPipeline": 0.3}},
                "scoreDetails": True,
            }
        },
        {"$project": {**_PROJECT, "scoreDetails": {"$meta": "scoreDetails"}}},
        {"$limit": limit},
    ]
    results = list(coll.aggregate(pipeline))
    for r in results:
        r["score"] = (r.get("scoreDetails") or {}).get("value", 0)
    return results


def _client_rrf_search(coll, codebase_id, query_text, query_vector, em, limit):
    pool = max(limit * 3, 20)

    def _run(pipeline):
        try:
            return list(coll.aggregate(pipeline))
        except OperationFailure as e:
            logger.warning(f"Search sub-pipeline failed: {getattr(e, 'details', e)}")
            return []

    vector_results = _run([
        _vector_stage(em, query_vector, codebase_id, pool),
        {"$project": _PROJECT},
    ])
    text_results = _run(_text_stage(query_text, codebase_id, pool) + [{"$project": _PROJECT}])

    fused = {}

    def _key(doc):
        return f"{doc.get('file_path')}::{doc.get('item_name')}"

    for rank, doc in enumerate(vector_results):
        k = _key(doc)
        entry = fused.setdefault(k, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (_RRF_K + rank + 1)
    for rank, doc in enumerate(text_results):
        k = _key(doc)
        entry = fused.setdefault(k, {"doc": doc, "score": 0.0})
        entry["score"] += 1.0 / (_RRF_K + rank + 1)

    ordered = sorted(fused.values(), key=lambda e: e["score"], reverse=True)[:limit]
    results = []
    for e in ordered:
        doc = e["doc"]
        doc["score"] = e["score"]
        doc["scoreDetails"] = {"value": e["score"]}
        results.append(doc)
    return results
