"""
src/workflow.py — LangGraph Workflow Engine
RAG-Based Customer Support Assistant

Graph Structure:
  input_node  ──►  [Conditional Router]
                        │
                ┌───────┴────────┐
            "normal"         "escalate"
                │                │
          process_node      hitl_node
                │                │
                └───────┬────────┘
                   output_node
"""

import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

try:
    from hitl import escalate_ticket
except ImportError:
    from src.hitl import escalate_ticket

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Groq LLM Setup ───────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"   # Fast, capable Groq model

# Initialise Groq LLM (lazily — avoids crash if key missing at import time)
_llm = None

def get_llm() -> ChatGroq:
    """Return (or create) the singleton Groq ChatLLM instance."""
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Please add it to your .env file."
            )
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=LLM_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )
        logger.info(f"Groq LLM initialised: {LLM_MODEL}")
    return _llm


# ─── Escalation Keywords ──────────────────────────────────────────────────────
ESCALATION_KEYWORDS = [
    "complaint", "refund", "urgent", "angry", "fraud",
    "manager", "furious", "stolen", "legal", "chargeback",
    "speak to human", "human agent", "disappointed", "unacceptable",
]


# ─── Graph State Schema ───────────────────────────────────────────────────────
class GraphState(TypedDict):
    """
    Shared state dictionary passed between all nodes in the LangGraph.
    Each node can read and update this state.
    """
    query           : str      # Original user query
    context         : str      # Retrieved ChromaDB context
    intent          : str      # "normal" | "escalate"
    response        : str      # Final answer shown to user
    ticket_id       : str      # Set if HITL is triggered
    requires_human  : bool     # Flag for HITL path


# ─── NODE 1: Input Node ───────────────────────────────────────────────────────
def input_node(state: GraphState) -> GraphState:
    """
    Entry node. Receives the query and performs a fast keyword-based
    intent classification before LLM-based routing.
    """
    query = state.get("query", "").strip()
    logger.info(f"[input_node] Received query: {query[:80]}...")

    # Fast keyword check (deterministic, zero latency)
    query_lower = query.lower()
    detected_intent = "escalate" if any(
        kw in query_lower for kw in ESCALATION_KEYWORDS
    ) else "normal"

    logger.info(f"[input_node] Keyword-based intent: {detected_intent}")

    return {
        **state,
        "query"  : query,
        "intent" : detected_intent,
    }


# ─── NODE 2a: Process Node (RAG Answer) ───────────────────────────────────────
def process_node(state: GraphState) -> GraphState:
    """
    RAG node. Uses retrieved context + Groq LLM to generate a grounded answer.
    The prompt explicitly constrains the LLM to the provided context only.
    """
    query   = state["query"]
    context = state.get("context", "No context available.")

    logger.info(f"[process_node] Generating RAG answer...")

    # ── Prompt Template ────────────────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a professional and empathetic customer support AI assistant. "
                "Your job is to answer user questions ONLY using the provided context below. "
                "If the context does not contain enough information to answer accurately, "
                "say: 'I'm sorry, I don't have enough information about that in our knowledge base. "
                "Please contact our support team for further assistance.' "
                "Do NOT make up answers. Keep responses concise, clear, and helpful.\n\n"
                "CONTEXT:\n{context}"
            ),
        ),
        ("human", "{query}"),
    ])

    try:
        llm    = get_llm()
        chain  = prompt | llm
        result = chain.invoke({"context": context, "query": query})
        answer = result.content.strip()

        logger.info("[process_node] Answer generated successfully.")

    except Exception as e:
        logger.error(f"[process_node] LLM call failed: {e}")
        answer = (
            "I apologize, but I'm experiencing a technical issue right now. "
            "Please try again in a moment or contact our support team directly."
        )

    return {
        **state,
        "response"      : answer,
        "requires_human": False,
        "intent"        : "normal",
    }


# ─── NODE 2b: HITL Node (Escalation) ─────────────────────────────────────────
def hitl_node(state: GraphState) -> GraphState:
    """
    Escalation node. Invokes the HITL module to generate a ticket
    and a user-facing escalation message.
    """
    query = state["query"]
    logger.info(f"[hitl_node] Escalating query to human agent...")

    result = escalate_ticket(
        query=query,
        chat_history=[{"role": "user", "content": query}],
    )

    return {
        **state,
        "response"      : result["response"],
        "ticket_id"     : result["ticket_id"],
        "requires_human": True,
        "intent"        : "escalate",
    }


# ─── NODE 3: Output Node ──────────────────────────────────────────────────────
def output_node(state: GraphState) -> GraphState:
    """
    Terminal node. Logs the final state and passes it through unchanged.
    In a production system this node would handle logging, analytics, etc.
    """
    logger.info(
        f"[output_node] Final intent={state['intent']} | "
        f"response_len={len(state.get('response', ''))}"
    )
    return state


# ─── Conditional Router ───────────────────────────────────────────────────────
def route_query(state: GraphState) -> str:
    """
    Edge routing function called after input_node.
    Returns the name of the next node to execute.

    Returns: "process_node" | "hitl_node"
    """
    intent = state.get("intent", "normal")
    logger.info(f"[router] Routing to → {'hitl_node' if intent == 'escalate' else 'process_node'}")
    return "hitl_node" if intent == "escalate" else "process_node"


# ─── Build the Compiled LangGraph ─────────────────────────────────────────────
def build_graph():
    """
    Assemble and compile the LangGraph StateGraph.

    Graph Topology:
        input_node
            ↓ (conditional edge via route_query)
        process_node ──────────────────────────┐
        hitl_node                              │
            ↓ (both converge here)             │
        output_node ◄──────────────────────────┘
            ↓
           END
    """
    graph = StateGraph(GraphState)

    # ── Register Nodes ─────────────────────────────────────────────────────
    graph.add_node("input_node",   input_node)
    graph.add_node("process_node", process_node)
    graph.add_node("hitl_node",    hitl_node)
    graph.add_node("output_node",  output_node)

    # ── Entry Point ────────────────────────────────────────────────────────
    graph.set_entry_point("input_node")

    # ── Conditional Edge: input_node → process_node OR hitl_node ──────────
    graph.add_conditional_edges(
        "input_node",
        route_query,
        {
            "process_node": "process_node",
            "hitl_node"   : "hitl_node",
        },
    )

    # ── Both branches converge at output_node ──────────────────────────────
    graph.add_edge("process_node", "output_node")
    graph.add_edge("hitl_node",    "output_node")

    # ── Terminal Edge ──────────────────────────────────────────────────────
    graph.add_edge("output_node", END)

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully.")
    return compiled


# ─── Module-Level Compiled Graph (singleton) ─────────────────────────────────
_compiled_graph = None

def get_graph():
    """Return (or build) the compiled LangGraph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─── Public Run Function ──────────────────────────────────────────────────────
def run_graph(query: str, context: str) -> dict:
    """
    Execute the LangGraph workflow for a single user query.

    Parameters
    ----------
    query   : User's question string.
    context : Relevant text retrieved from ChromaDB.

    Returns
    -------
    dict — Final GraphState with 'response' and 'intent' populated.
    """
    initial_state: GraphState = {
        "query"          : query,
        "context"        : context,
        "intent"         : "normal",    # Will be updated by input_node
        "response"       : "",
        "ticket_id"      : "",
        "requires_human" : False,
    }

    try:
        graph        = get_graph()
        final_state  = graph.invoke(initial_state)
        return final_state

    except Exception as e:
        logger.error(f"[run_graph] LangGraph execution failed: {e}")
        return {
            **initial_state,
            "response" : (
                "An internal error occurred while processing your request. "
                "Please try again or contact our support team."
            ),
            "intent"   : "error",
        }
