"""
Follow-up QA node — answers user questions about an existing report
without re-running the full pipeline.

Uses: existing report + findings + analysis_results + knowledge retrieval
+ optional targeted analysis tools (no pipeline re-run).
"""
from __future__ import annotations
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, MAX_RETRIEVAL_DOCS
from fraud_analytics.knowledge.vector_store import FraudKnowledgeBase
from fraud_analytics.config import VECTOR_STORE_PATH
from fraud_analytics.tools.pipeline import run_pipeline
from fraud_analytics.tools.analysis import (
    analyze_fraud_monthly, analyze_fraud_weekly,
    analyze_promo_weekly, analyze_coin2dd, analyze_appid_breakdown,
)

# ── Light tools available to followup (no full pipeline re-run) ───────────────

@tool
def get_fraud_monthly_detail() -> list:
    """Get detailed MoM analysis for all months in the fraud_monthly_loss table."""
    r = run_pipeline()
    return analyze_fraud_monthly(r.get("fraud_monthly_loss", []))

@tool
def get_fraud_weekly_detail() -> list:
    """Get detailed WoW analysis for the latest weeks in the fraud_weekly_loss table."""
    r = run_pipeline()
    return analyze_fraud_weekly(r.get("fraud_weekly_loss", []))

@tool
def get_promo_detail() -> list:
    """Get detailed promo abuse analysis for the latest weeks."""
    r = run_pipeline()
    return analyze_promo_weekly(r.get("promo_weekly_abuse", []))

@tool
def get_coin2dd_detail() -> list:
    """Get detailed Coin2DD abuse analysis for all months."""
    r = run_pipeline()
    return analyze_coin2dd(r.get("coin2dd_monthly", []))

@tool
def get_appid_detail() -> list:
    """Get top appID fraud breakdown with MoM comparison."""
    r = run_pipeline()
    return analyze_appid_breakdown(r.get("appid_fraud_breakdown", []))

@tool
def get_raw_table(table_name: str) -> list:
    """Get the raw records for a specific output table.
    table_name must be one of: fraud_monthly_loss, fraud_weekly_loss,
    promo_weekly_abuse, coin2dd_monthly, appid_fraud_breakdown."""
    r = run_pipeline()
    data = r.get(table_name, [])
    return data[-10:] if data else []  # return last 10 rows

_FOLLOWUP_TOOLS = [
    get_fraud_monthly_detail,
    get_fraud_weekly_detail,
    get_promo_detail,
    get_coin2dd_detail,
    get_appid_detail,
    get_raw_table,
]
_TOOL_MAP = {t.name: t for t in _FOLLOWUP_TOOLS}

_SYSTEM = """You are a ZaloPay Risk Analyst answering a follow-up question about a fraud report.

Answer the question using:
1. The existing report and findings in context (primary source)
2. Retrieved domain knowledge (for thresholds, patterns, terminology)
3. Call a tool ONLY if the answer requires live data not already in the report

Answer rules:
- Be concise and direct — 3-8 sentences unless a table or list is needed
- Always cite specific numbers from the report or tool output
- If you reference a pattern (e.g. Campaign Splitting), explain it briefly
- Use ZaloPay priority labels (CRITICAL / ALERT / WATCH / STABLE) when relevant
- If the question cannot be answered from available data, say so clearly

Context available:
EXISTING REPORT (excerpt):
{report_excerpt}

FINDINGS:
{findings}

ANALYSIS RESULTS:
{analysis}

DOMAIN KNOWLEDGE:
{knowledge}

Current report scope: {report_type} | {fraud_pillar} | {date_range}
"""

_kb = None


def _get_kb() -> FraudKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = FraudKnowledgeBase(persist_path=VECTOR_STORE_PATH)
    return _kb


def followup_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(_FOLLOWUP_TOOLS)

    question = state.get("user_request", "")
    final_report = state.get("final_report") or ""
    findings = state.get("findings") or []
    analysis = state.get("analysis_results") or {}
    dr = state.get("date_range", {})

    # Retrieve relevant knowledge for the question
    kb = _get_kb()
    docs = kb.search(question, k=MAX_RETRIEVAL_DOCS)
    knowledge = "\n\n".join(
        f"[{d.get('metadata', {}).get('source', '?')}] {d['content'][:400]}"
        for d in docs[:4]
    )

    try:
        findings_str  = json.dumps(findings, indent=2, default=str)[:1500]
        analysis_str  = json.dumps(analysis, indent=2, default=str)[:1000]
    except Exception:
        findings_str  = str(findings)[:1500]
        analysis_str  = str(analysis)[:1000]

    # Trim report to most relevant part (avoid huge context)
    report_excerpt = final_report[:2000] if final_report else "No report available."

    system_content = _SYSTEM.format(
        report_excerpt=report_excerpt,
        findings=findings_str,
        analysis=analysis_str,
        knowledge=knowledge or "No additional knowledge retrieved.",
        report_type=state.get("report_type", "N/A"),
        fraud_pillar=state.get("fraud_pillar", "N/A"),
        date_range=f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}",
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question),
    ]

    # Allow up to 3 tool calls for data lookup
    for _ in range(4):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            tool_fn = _TOOL_MAP.get(tc["name"])
            try:
                result = tool_fn.invoke(tc["args"]) if tool_fn else f"Unknown tool: {tc['name']}"
            except Exception as exc:
                result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    answer = response.content.strip() if hasattr(response, "content") else "I couldn't answer that question."

    history = list(state.get("conversation_history") or [])
    history.append({"role": "assistant", "content": answer})

    return {
        "agent_message": answer,
        "conversation_history": history,
        "messages": state.get("messages", []) + [{
            "role": "followup",
            "content": f"Answered follow-up: {question[:80]}",
        }],
    }
