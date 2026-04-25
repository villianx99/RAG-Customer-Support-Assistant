"""
src/hitl.py — Human-in-the-Loop (HITL) Escalation Module
RAG-Based Customer Support Assistant
"""

import os
import json
import random
import logging
from datetime import datetime

# ─── Module Logger ────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Ticket Log File Path ─────────────────────────────────────────────────────
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "..", "logs")
TICKET_FILE = os.path.join(LOGS_DIR, "tickets.json")


def _ensure_log_dir():
    """Make sure the logs directory exists."""
    os.makedirs(LOGS_DIR, exist_ok=True)


def _generate_ticket_id() -> str:
    """
    Generate a unique support ticket ID in the format TKT-YYYY-XXXXX.
    Example: TKT-2024-84291
    """
    year   = datetime.now().year
    serial = random.randint(10000, 99999)
    return f"TKT-{year}-{serial}"


def _determine_priority(query: str) -> str:
    """
    Assign a priority level to the escalated ticket based on trigger keywords.
    - CRITICAL : fraud, legal, stolen
    - HIGH      : urgent, refund, angry
    - MEDIUM    : complaint, manager
    """
    query_lower = query.lower()

    critical_keywords = ["fraud", "stolen", "legal", "chargeback"]
    high_keywords     = ["urgent", "refund", "angry", "furious", "immediately"]
    medium_keywords   = ["complaint", "manager", "disappointed"]

    if any(kw in query_lower for kw in critical_keywords):
        return "CRITICAL"
    elif any(kw in query_lower for kw in high_keywords):
        return "HIGH"
    elif any(kw in query_lower for kw in medium_keywords):
        return "MEDIUM"
    else:
        return "HIGH"   # Default escalation priority


def _persist_ticket(ticket: dict):
    """
    Save the ticket payload to a local JSON log file for the human agent dashboard.
    Appends to existing tickets rather than overwriting.
    """
    _ensure_log_dir()

    existing_tickets = []

    # Load existing tickets if the file exists
    if os.path.exists(TICKET_FILE):
        try:
            with open(TICKET_FILE, "r", encoding="utf-8") as f:
                existing_tickets = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_tickets = []   # Reset on corruption

    existing_tickets.append(ticket)

    with open(TICKET_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_tickets, f, indent=4, ensure_ascii=False)

    logger.info(f"Ticket {ticket['ticket_id']} persisted to {TICKET_FILE}")


def escalate_ticket(query: str, chat_history: list = None) -> dict:
    """
    Main escalation function.

    Triggered when the LangGraph conditional router determines the user's
    query requires human intervention (anger, fraud, refund, complaint, etc.).

    Parameters
    ----------
    query        : The user's raw query string.
    chat_history : Optional list of prior messages for agent context.

    Returns
    -------
    dict with:
        - ticket_id   : Unique ticket identifier
        - priority    : CRITICAL / HIGH / MEDIUM
        - message     : User-facing response string
        - response    : Same as message (for graph state compatibility)
        - intent      : "escalate"
        - timestamp   : ISO timestamp
    """
    ticket_id = _generate_ticket_id()
    priority  = _determine_priority(query)
    timestamp = datetime.now().isoformat()

    # ── Build the ticket payload ────────────────────────────────────────────
    ticket = {
        "ticket_id"   : ticket_id,
        "priority"    : priority,
        "user_query"  : query,
        "chat_history": chat_history or [],
        "status"      : "OPEN",
        "assigned_to" : "Unassigned",
        "timestamp"   : timestamp,
        "notes"       : "Auto-escalated by AI Support Assistant via LangGraph HITL node.",
    }

    # ── Persist to logs/ ───────────────────────────────────────────────────
    _persist_ticket(ticket)

    # ── Compose user-facing escalation message ─────────────────────────────
    user_message = (
        f"⚠️ **Escalated to Human Agent** | Ticket `{ticket_id}`\n\n"
        f"I understand this is a **{priority} priority** matter. "
        f"I have raised a support ticket and a human agent has been notified.\n\n"
        f"🎫 **Ticket ID:** `{ticket_id}`\n"
        f"🔺 **Priority:** `{priority}`\n"
        f"🕐 **Raised At:** `{datetime.now().strftime('%d %b %Y, %I:%M %p')}`\n\n"
        f"Our team will reach out to you shortly. "
        f"Please keep your Ticket ID handy for follow-up."
    )

    logger.info(
        f"HITL Escalation → Ticket: {ticket_id} | Priority: {priority} | "
        f"Query: {query[:60]}..."
    )

    return {
        "ticket_id" : ticket_id,
        "priority"  : priority,
        "message"   : user_message,
        "response"  : user_message,    # aligned with GraphState key
        "intent"    : "escalate",
        "timestamp" : timestamp,
    }


def get_all_tickets() -> list:
    """
    Utility function to retrieve all logged tickets.
    Useful for building an admin dashboard in future enhancements.
    """
    if not os.path.exists(TICKET_FILE):
        return []

    try:
        with open(TICKET_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Could not read ticket log: {e}")
        return []
