"""
Fraud Analytics Agent — HTTP Server
====================================
FastAPI wrapper for AgentBase Runtime deployment.

Endpoints:
  GET  /health          → liveness probe (AgentBase requires this)
  POST /chat            → send a message; start or continue a session
  DELETE /chat/{id}     → explicitly close a session
"""
from __future__ import annotations

import os
import uuid
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Fraud Analytics Agent", version="1.0.0")

# ── Session store ─────────────────────────────────────────────────────────────
# Tracks per-session state: whether the first invoke has been done and the last
# seen report (so we only surface a report once per analysis run).
_sessions: Dict[str, Dict[str, Any]] = {}


def _get_or_create_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {"started": False, "seen_report": ""}
    return _sessions[session_id]


# ── Startup: warm up the graph and vector store ───────────────────────────────

@app.on_event("startup")
async def startup():
    log.info("Warming up fraud analytics graph ...")
    from fraud_analytics.graph.fraud_graph import get_chat_graph
    get_chat_graph()
    log.info("Graph ready.")


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str           # agent question / follow-up / farewell
    report: str             # non-empty only when a new report was generated
    done: bool              # True when session has ended


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe required by AgentBase Runtime."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message to the fraud analytics agent.

    - Omit `session_id` on the first call; the server returns a new one.
    - Pass the returned `session_id` on every subsequent call in the same conversation.
    - When `done` is True the session is closed and the ID is no longer valid.
    """
    from fraud_analytics.graph.fraud_graph import get_chat_graph, make_chat_config
    from langgraph.types import Command

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_or_create_session(session_id)
    graph = get_chat_graph()
    config = make_chat_config(session_id)

    try:
        if not session["started"]:
            initial_state = {
                "user_request": req.message,
                "conversation_history": [{"role": "user", "content": req.message}],
                "retry_count": 0,
                "messages": [],
            }
            state = graph.invoke(initial_state, config)
            session["started"] = True
        else:
            state = graph.invoke(Command(resume=req.message), config)

    except Exception as exc:
        log.exception("Graph error for session %s", session_id)
        _sessions.pop(session_id, None)
        raise HTTPException(status_code=500, detail=str(exc))

    snapshot = graph.get_state(config)
    done = not snapshot.next

    # Surface a new report only once
    report = state.get("final_report") or ""
    new_report = ""
    if report and report != session["seen_report"]:
        new_report = report
        session["seen_report"] = report

    agent_message = snapshot.values.get("agent_message") or ""
    if done and not agent_message:
        agent_message = state.get("agent_message") or "Session ended."

    if done:
        _sessions.pop(session_id, None)

    return ChatResponse(
        session_id=session_id,
        response=agent_message,
        report=new_report,
        done=done,
    )


@app.delete("/chat/{session_id}")
def close_session(session_id: str):
    """Explicitly close and discard a session."""
    _sessions.pop(session_id, None)
    return {"session_id": session_id, "closed": True}


@app.get("/sessions")
def list_sessions():
    """Debug endpoint — lists active session IDs."""
    return {"active_sessions": list(_sessions.keys())}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
