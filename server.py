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
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

_executor = ThreadPoolExecutor(max_workers=10)
SERVER_TIMEOUT: int = int(os.getenv("SERVER_TIMEOUT_SECONDS", "150"))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Fraud Analytics Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    log.info("Initializing knowledge base ...")
    from fraud_analytics.knowledge.vector_store import FraudKnowledgeBase
    kb = FraudKnowledgeBase()
    # Build vector store from txt docs if not already persisted
    if not os.path.exists(os.path.join(kb.persist_path, "index.faiss")):
        n = kb.rebuild()
        log.info(f"Vector store built from {n} documents.")
    else:
        log.info("Vector store loaded from disk.")
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

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    """Liveness probe required by AgentBase Runtime."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a message to the fraud analytics agent.

    - Omit `session_id` on the first call; the server returns a new one.
    - Pass the returned `session_id` on every subsequent call in the same conversation.
    - When `done` is True the session is closed and the ID is no longer valid.
    """
    from fraud_analytics.graph.fraud_graph import get_chat_graph, make_chat_config
    from langgraph.types import Command

    _QUIT_WORDS = {"exit", "quit", "bye", "goodbye", "stop", "end", "no", "done", "thoát", "kết thúc"}

    session_id = req.session_id or str(uuid.uuid4())
    session = _get_or_create_session(session_id)

    # Short-circuit quit keywords on an active session — no LLM call needed
    if session["started"] and req.message.strip().lower() in _QUIT_WORDS:
        _sessions.pop(session_id, None)
        return ChatResponse(
            session_id=session_id,
            response="Session ended. Goodbye!",
            report="",
            done=True,
        )

    graph = get_chat_graph()
    config = make_chat_config(session_id)

    def _invoke():
        if not session["started"]:
            initial_state = {
                "user_request": req.message,
                "conversation_history": [{"role": "user", "content": req.message}],
                "retry_count": 0,
                "messages": [],
            }
            result = graph.invoke(initial_state, config)
            session["started"] = True
            return result
        return graph.invoke(Command(resume=req.message), config)

    try:
        loop = asyncio.get_event_loop()
        state = await asyncio.wait_for(
            loop.run_in_executor(_executor, _invoke),
            timeout=SERVER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.warning("Session %s timed out after %ds", session_id, SERVER_TIMEOUT)
        _sessions.pop(session_id, None)
        return ChatResponse(
            session_id=session_id,
            response=f"⏱️ Request timed out after {SERVER_TIMEOUT}s. Please try a simpler request.",
            report="",
            done=True,
        )
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

    # When a report was just generated but no agent message is set, surface a brief prompt
    if new_report and not agent_message:
        agent_message = "✅ Report generated above. Ask a follow-up question or request another report."

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


def _agentbase_headers() -> dict:
    """Build auth headers for AgentBase Memory REST API using user IAM credentials.

    Reads AGENTBASE_IAM_CLIENT_ID / AGENTBASE_IAM_CLIENT_SECRET from env first —
    these are dedicated vars that the runtime does NOT auto-inject, so they always
    hold the real user credentials even when GREENNODE_CLIENT_ID is overridden by
    the runtime service-account injection.
    Falls back to .greennode.json for local dev.
    """
    import httpx

    client_id     = os.getenv("AGENTBASE_IAM_CLIENT_ID", "")
    client_secret = os.getenv("AGENTBASE_IAM_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        # Local dev fallback — .greennode.json not present in deployed containers
        import json as _json
        try:
            with open(".greennode.json") as f:
                creds = _json.load(f)
            client_id     = creds.get("client_id", "")
            client_secret = creds.get("client_secret", "")
        except Exception:
            return {}

    token_url = "https://iamapis.vngcloud.vn/accounts-api/v2/auth/token"
    resp = httpx.post(token_url, data={
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }, timeout=10)
    resp.raise_for_status()
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _decode_memory_events(events: list) -> dict:
    """Decode AgentBase memory blob events → conversation_history + final_report."""
    import base64, msgpack as _mp

    conv_events   = []
    report_events = []

    for e in events:
        b64 = e.get("payload", {}).get("binaryData", "")
        if not b64:
            continue
        try:
            parsed = __import__("json").loads(base64.b64decode(b64).decode("utf-8", errors="replace"))
            if parsed.get("event_type") != "channel_data":
                continue
            ch  = parsed.get("channel", "")
            val = parsed.get("value", {})
            if not (isinstance(val, dict) and val.get("type") == "msgpack"):
                continue
            decoded = _mp.unpackb(base64.b64decode(val["data"]), raw=False)
            ts = e.get("eventTimestamp", "")
            if ch == "conversation_history":
                conv_events.append((ts, decoded))
            elif ch == "final_report":
                report_events.append((ts, decoded))
        except Exception:
            pass

    conv_events.sort(key=lambda x: x[0])
    report_events.sort(key=lambda x: x[0])

    history = conv_events[-1][1]  if conv_events  else []
    report  = report_events[-1][1] if report_events else ""

    if not isinstance(history, list):
        history = []
    if not isinstance(report, str):
        report = ""

    return {"history": history, "has_report": bool(report), "report": report}


@app.get("/conversations")
async def list_conversations():
    """List all sessions across all actors in AgentBase memory."""
    memory_id = os.getenv("AGENTBASE_MEMORY_ID", "")
    if not memory_id:
        return JSONResponse({"sessions": []})

    def _fetch():
        import httpx
        headers = _agentbase_headers()
        if not headers:
            return []
        base = "https://agentbase.api.vngcloud.vn/memory"

        # 1. List all actors
        r = httpx.get(f"{base}/memories/{memory_id}/actors", headers=headers, timeout=10)
        r.raise_for_status()
        actors = [a["actorId"] for a in (r.json().get("listData") or [])]

        # 2. Collect sessions for each actor
        sessions = []
        for actor in actors:
            r2 = httpx.get(
                f"{base}/memories/{memory_id}/actors/{actor}/sessions",
                params={"page": 1, "size": 100},
                headers=headers, timeout=10,
            )
            r2.raise_for_status()
            for s in r2.json().get("listData") or []:
                sid = s.get("sessionId", "")
                if sid and not any(x["session_id"] == sid for x in sessions):
                    sessions.append({"session_id": sid, "actor_id": actor})
        return sessions

    try:
        loop = asyncio.get_event_loop()
        sessions = await asyncio.wait_for(loop.run_in_executor(_executor, _fetch), timeout=20)
        return JSONResponse({"sessions": sessions})
    except Exception as exc:
        log.warning("list_conversations failed: %s", exc)
        return JSONResponse({"sessions": []})


@app.get("/conversations/{session_id}/history")
async def get_conversation_history(session_id: str, actor_id: str = ""):
    """Decode conversation history directly from AgentBase memory events."""
    memory_id = os.getenv("AGENTBASE_MEMORY_ID", "")

    # Fallback: read from LangGraph in-process graph (for active HTTP sessions)
    if not memory_id:
        def _local():
            from fraud_analytics.graph.fraud_graph import get_chat_graph, make_chat_config
            graph  = get_chat_graph()
            config = make_chat_config(session_id, actor_id or None)
            state  = graph.get_state(config)
            if not state or not state.values:
                return {"history": [], "has_report": False, "report": ""}
            history = state.values.get("conversation_history") or []
            report  = state.values.get("final_report") or ""
            return {"history": history, "has_report": bool(report), "report": report}
        try:
            loop = asyncio.get_event_loop()
            data = await asyncio.wait_for(loop.run_in_executor(_executor, _local), timeout=30)
            return JSONResponse(data)
        except Exception as exc:
            log.warning("get_conversation_history(%s) local failed: %s", session_id, exc)
            return JSONResponse({"history": [], "has_report": False, "report": ""})

    def _fetch_events():
        import httpx
        headers = _agentbase_headers()
        if not headers:
            return []
        base    = "https://agentbase.api.vngcloud.vn/memory"

        # Determine actor — use provided actor_id or try all actors
        actors_to_try = [actor_id] if actor_id else []
        if not actors_to_try:
            r = httpx.get(f"{base}/memories/{memory_id}/actors", headers=headers, timeout=10)
            r.raise_for_status()
            actors_to_try = [a["actorId"] for a in (r.json().get("listData") or [])]

        for actor in actors_to_try:
            r2 = httpx.get(
                f"{base}/memories/{memory_id}/actors/{actor}/sessions/{session_id}/events",
                params={"page": 1, "size": 200},
                headers=headers, timeout=15,
            )
            if r2.status_code == 200:
                events = r2.json().get("listData") or []
                if events:
                    return events
        return []

    try:
        loop = asyncio.get_event_loop()
        events = await asyncio.wait_for(loop.run_in_executor(_executor, _fetch_events), timeout=25)
        data   = _decode_memory_events(events)
        return JSONResponse(data)
    except Exception as exc:
        log.warning("get_conversation_history(%s) failed: %s", session_id, exc)
        return JSONResponse({"history": [], "has_report": False, "report": ""})


# ── Microsoft Teams Bot Framework endpoint ────────────────────────────────────

@app.get("/api/messages")
async def teams_messages_verify():
    """Endpoint verification probe from Azure Bot / Teams."""
    return {"status": "ok"}


@app.post("/api/messages")
async def teams_messages(req: Request):
    """Bot Framework endpoint for Microsoft Teams channel."""
    from botbuilder.core import BotFrameworkAdapterSettings, BotFrameworkAdapter, TurnContext
    from botbuilder.schema import Activity

    app_id     = os.getenv("MicrosoftAppId", "")
    app_secret = os.getenv("MicrosoftAppPassword", "")

    adapter = BotFrameworkAdapter(BotFrameworkAdapterSettings(app_id, app_secret))

    body        = await req.json()
    activity    = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    async def on_turn(turn_context: TurnContext):
        if turn_context.activity.type != "message":
            return

        user_text  = (turn_context.activity.text or "").strip()
        session_id = turn_context.activity.conversation.id

        session = _get_or_create_session(session_id)

        from fraud_analytics.graph.fraud_graph import get_chat_graph, make_chat_config
        from langgraph.types import Command

        graph  = get_chat_graph()
        config = make_chat_config(session_id)

        def _invoke():
            if not session["started"]:
                initial_state = {
                    "user_request": user_text,
                    "conversation_history": [{"role": "user", "content": user_text}],
                    "retry_count": 0,
                    "messages": [],
                }
                result = graph.invoke(initial_state, config)
                session["started"] = True
                return result
            return graph.invoke(Command(resume=user_text), config)

        try:
            loop = asyncio.get_event_loop()
            state = await asyncio.wait_for(
                loop.run_in_executor(_executor, _invoke),
                timeout=SERVER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            _sessions.pop(session_id, None)
            await turn_context.send_activity(f"⏱️ Request timed out after {SERVER_TIMEOUT}s.")
            return
        except Exception as exc:
            log.exception("Teams graph error session=%s", session_id)
            await turn_context.send_activity(f"❌ Error: {exc}")
            return

        snapshot   = graph.get_state(config)
        done       = not snapshot.next
        report     = state.get("final_report") or ""
        new_report = ""
        if report and report != session["seen_report"]:
            new_report = report
            session["seen_report"] = report

        agent_message = snapshot.values.get("agent_message") or ""
        if done and not agent_message:
            agent_message = state.get("agent_message") or "Session ended."
        if new_report and not agent_message:
            agent_message = "✅ Report generated. Ask a follow-up or request another report."

        if done:
            _sessions.pop(session_id, None)

        # Send report first, then the follow-up message
        if new_report:
            await turn_context.send_activity(new_report[:4000])  # Teams msg limit ~28k chars
        if agent_message:
            await turn_context.send_activity(agent_message)

    from fastapi.responses import Response
    try:
        await adapter.process_activity(activity, auth_header, on_turn)
        return Response(status_code=200)
    except PermissionError as exc:
        log.warning("Teams auth rejected: %s", exc)
        return Response(status_code=401)
    except Exception as exc:
        log.exception("Teams /api/messages error: %s", exc)
        return Response(status_code=500)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
