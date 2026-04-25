"""
src/main.py — Application Orchestrator
RAG-Based Customer Support Assistant

Responsibilities:
  - process_pdf()  : Runs the full PDF ingestion pipeline
  - ask_query()    : Retrieves context + runs LangGraph + returns final answer
"""

import os
import sys
import logging

# ─── Ensure src/ is importable when called from project root ─────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils     import setup_logger
from retriever import run_ingestion_pipeline, create_embeddings, load_chromadb, retrieve_context
from workflow  import run_graph

# ─── Module Logger ────────────────────────────────────────────────────────────
logger = setup_logger("main")

# ─── Persistent Embedding Model & Vector DB ───────────────────────────────────
# These are loaded once and reused to avoid re-initialisation on every query.
_embeddings = None
_vectordb   = None


def _get_vectordb():
    """
    Lazy-load and cache the embedding model + ChromaDB vector store.
    Called at query time, not at import time, to keep startup fast.
    """
    global _embeddings, _vectordb

    if _embeddings is None:
        logger.info("Initialising embedding model...")
        _embeddings = create_embeddings()

    if _vectordb is None:
        logger.info("Loading ChromaDB from disk...")
        _vectordb = load_chromadb(_embeddings)

    return _vectordb


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def process_pdf(file_path: str) -> dict:
    """
    Full ingestion pipeline entry point.
    Called by app.py when the user clicks 'Process PDF'.

    Steps:
      1. Validate file path
      2. Run ingestion pipeline (load → chunk → embed → store)
      3. Reset cached vectordb so fresh data is used on next query

    Parameters
    ----------
    file_path : Absolute path to the uploaded PDF file.

    Returns
    -------
    dict — { "status": "success"|"error", "message": str }
    """
    global _vectordb, _embeddings

    logger.info(f"process_pdf called: {file_path}")

    if not file_path or not os.path.exists(file_path):
        return {"status": "error", "message": "PDF file path is invalid or file does not exist."}

    # Run full pipeline
    result = run_ingestion_pipeline(file_path)

    # Reset the cached DB so the next query uses the newly ingested data
    _vectordb = None

    logger.info(f"process_pdf result: {result['status']} — {result['message']}")
    return result


def ask_query(query: str) -> dict:
    """
    Query answering entry point.
    Called by app.py when the user submits a question.

    Steps:
      1. Validate query
      2. Load ChromaDB (from disk cache)
      3. Retrieve semantic context
      4. Run LangGraph workflow (auto-routes to RAG or HITL)
      5. Return final state dict

    Parameters
    ----------
    query : The user's natural language question.

    Returns
    -------
    dict — {
        "response" : str   (answer or escalation message),
        "intent"   : str   ("normal" | "escalate" | "error"),
        "ticket_id": str   (populated only on HITL escalation),
    }
    """
    logger.info(f"ask_query called: {query[:80]}...")

    # ── Input validation ────────────────────────────────────────────────────
    if not query or not query.strip():
        return {
            "response" : "Please enter a valid question.",
            "intent"   : "error",
            "ticket_id": "",
        }

    if len(query) > 2000:
        return {
            "response" : "Your query is too long. Please shorten it and try again.",
            "intent"   : "error",
            "ticket_id": "",
        }

    try:
        # ── Step 1: Load vector store ───────────────────────────────────────
        vectordb = _get_vectordb()

        # ── Step 2: Retrieve relevant context ──────────────────────────────
        context = retrieve_context(query, vectordb)
        logger.info(f"Context retrieved ({len(context)} chars).")

        # ── Step 3: Run LangGraph ───────────────────────────────────────────
        final_state = run_graph(query=query, context=context)

        return {
            "response" : final_state.get("response", "No response generated."),
            "intent"   : final_state.get("intent", "normal"),
            "ticket_id": final_state.get("ticket_id", ""),
        }

    except FileNotFoundError:
        logger.error("ChromaDB not found — PDF may not have been processed yet.")
        return {
            "response" : (
                "No knowledge base found. Please upload and process a PDF document first."
            ),
            "intent"   : "error",
            "ticket_id": "",
        }

    except Exception as e:
        logger.error(f"ask_query error: {e}")
        return {
            "response" : (
                "An unexpected error occurred while processing your query. "
                "Please try again."
            ),
            "intent"   : "error",
            "ticket_id": "",
        }
