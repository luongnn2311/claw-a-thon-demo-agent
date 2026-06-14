from __future__ import annotations
import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from fraud_analytics.state import FraudReportState
from fraud_analytics.agents.orchestrator import orchestrator_node
from fraud_analytics.agents.retrieval import retrieval_node
from fraud_analytics.agents.query import query_node
from fraud_analytics.agents.summarizer import summarizer_node
from fraud_analytics.agents.reasoning import reasoning_node
from fraud_analytics.agents.validation import validation_node
from fraud_analytics.agents.report import report_node
from fraud_analytics.config import MAX_VALIDATION_RETRIES
from fraud_analytics.utils.circuit_breaker import check_circuit_breaker


# ── Node visit tracker ────────────────────────────────────────────────────────

def _tracked(node_fn):
    """Wrap a node to increment the visit counter."""
    def wrapper(state: FraudReportState) -> Dict[str, Any]:
        result = node_fn(state)
        result["total_node_visits"] = state.get("total_node_visits", 0) + 1
        return result
    wrapper.__name__ = node_fn.__name__
    return wrapper


# ── Shared routing ────────────────────────────────────────────────────────────

def _route_after_validation(state: FraudReportState) -> str:
    if check_circuit_breaker(state):
        return "report"
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_VALIDATION_RETRIES:
        return "report"
    validation = state.get("validation_result") or {}
    if validation.get("validated", False):
        return "report"
    next_step = validation.get("next_step", "report")
    if next_step in ("query", "retrieval"):
        return next_step
    return "report"


def _route_conversation(state: FraudReportState) -> str:
    action = state.get("next_action", "proceed")
    # Guard: unknown actions fall back to proceed
    if action not in ("proceed", "clarify", "follow_up", "answer", "end"):
        return "proceed"
    return action


# ── Human-in-the-loop node ────────────────────────────────────────────────────

def _human_input_node(state: FraudReportState) -> Dict[str, Any]:
    """Pause the graph and wait for the user's next message via interrupt()."""
    from langgraph.types import interrupt

    user_message: str = interrupt(state.get("agent_message", ""))

    history = list(state.get("conversation_history") or [])
    history.append({"role": "user", "content": user_message})

    return {
        "user_request": user_message,
        "conversation_history": history,
    }


# ── Shared pipeline helper ────────────────────────────────────────────────────

def _add_pipeline(graph: StateGraph) -> None:
    """Register analysis pipeline nodes and their fixed edges."""
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("retrieval", _tracked(retrieval_node))
    graph.add_node("query", _tracked(query_node))
    graph.add_node("summarizer", _tracked(summarizer_node))
    graph.add_node("reasoning", _tracked(reasoning_node))
    graph.add_node("validation", _tracked(validation_node))
    graph.add_node("report", report_node)

    graph.add_edge("orchestrator", "retrieval")
    graph.add_edge("retrieval", "query")
    graph.add_edge("query", "summarizer")
    graph.add_edge("summarizer", "reasoning")
    graph.add_edge("reasoning", "validation")

    graph.add_conditional_edges(
        "validation",
        _route_after_validation,
        {"query": "query", "retrieval": "retrieval", "report": "report"},
    )


# ── Single-run graph (backward compat) ───────────────────────────────────────

def build_fraud_graph():
    """
    Single-shot pipeline: one request → one report → END.

    orchestrator → retrieval → query → summarizer → reasoning → validation → report → END
    """
    graph = StateGraph(FraudReportState)
    _add_pipeline(graph)
    graph.set_entry_point("orchestrator")
    graph.add_edge("report", END)
    return graph.compile()


# ── Multi-turn conversational graph ──────────────────────────────────────────

def _build_chat_graph(checkpointer):
    """Compile the conversational graph against an existing checkpointer."""
    from fraud_analytics.agents.conversation import conversation_node
    from fraud_analytics.agents.followup import followup_node

    graph = StateGraph(FraudReportState)
    graph.add_node("conversation", conversation_node)
    graph.add_node("human_input", _human_input_node)
    graph.add_node("followup", followup_node)
    _add_pipeline(graph)
    graph.set_entry_point("conversation")

    graph.add_conditional_edges(
        "conversation",
        _route_conversation,
        {
            "proceed":    "orchestrator",
            "clarify":    "human_input",
            "follow_up":  "human_input",
            "answer":     "followup",
            "end":        END,
        },
    )
    graph.add_edge("human_input", "conversation")
    graph.add_edge("followup", "human_input")
    # Always pause at human_input after a report.
    graph.add_edge("report", "human_input")

    return graph.compile(checkpointer=checkpointer)


# Singleton — one compiled graph, one MemorySaver, many thread IDs (sessions)
_chat_graph = None
_chat_checkpointer = None


def get_chat_graph():
    """
    Return the shared compiled chat graph and its checkpointer.
    Use make_chat_config(session_id) to get a per-session config.
    """
    global _chat_graph, _chat_checkpointer
    if _chat_graph is None:
        from langgraph.checkpoint.memory import MemorySaver
        _chat_checkpointer = MemorySaver()
        _chat_graph = _build_chat_graph(_chat_checkpointer)
    return _chat_graph


def make_chat_config(session_id: str) -> dict:
    """Return a LangGraph config scoped to a specific session / thread."""
    return {"configurable": {"thread_id": session_id}}


def build_chat_graph():
    """
    Backward-compatible wrapper used by main.py (single CLI session).
    Returns (compiled_graph, config).
    """
    graph = get_chat_graph()
    config = make_chat_config("fraud-chat-cli")
    return graph, config
