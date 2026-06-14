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
from fraud_analytics.knowledge.web_enrichment import search_web

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

@tool
def search_fintech_web(query: str) -> list:
    """Search the internet for fintech / payment fraud risk knowledge.

    Call this when:
    - The user explicitly asks to search the web / look online, OR
    - The question is about an industry concept, regulation, or attack
      technique and the local knowledge does NOT adequately cover it.

    Do NOT call if the local knowledge already fully answers the question.
    Use web results as supplementary background context — synthesise them
    into your answer, never quote them verbatim.

    Args:
        query: concise search phrase (e.g. "3DS2 liability shift card fraud",
               "CNP fraud prevention techniques", "VAMP Visa acquirer program")
    """
    return search_web(query, max_results=3)

_FOLLOWUP_TOOLS = [
    get_fraud_monthly_detail,
    get_fraud_weekly_detail,
    get_promo_detail,
    get_coin2dd_detail,
    get_appid_detail,
    get_raw_table,
    search_fintech_web,
]
_TOOL_MAP = {t.name: t for t in _FOLLOWUP_TOOLS}

_SYSTEM = """You are a ZaloPay Fraud Analytics Assistant — a domain expert on ZaloPay fraud, risk, and promo abuse.

You help with TWO types of questions:
  A. General questions — domain concepts, ZaloPay terminology, fraud patterns, thresholds,
     team ownership, "what is X", "explain Y", "how does Z work"
  B. Data / report questions — specific numbers from ZaloPay data, follow-up on an existing report

TOOL USAGE RULES — follow strictly:
1. data tools (get_fraud_monthly_detail, get_raw_table, etc.)
   → call ONLY when the question asks for specific numbers/data not already available in context
   → for general/concept questions, do NOT call data tools
2. search_fintech_web
   → call when:
      a. The user explicitly asks to search / look online, OR
      b. The question is about an industry concept not covered by local knowledge
   → do NOT call if the local knowledge already fully answers the question
   → Use web results as background context to enrich your answer — never quote them directly

ANSWER RULES:
- Be concise — 3-8 sentences unless a table is genuinely needed
- For concept/domain questions, answer from domain knowledge — no data tools needed
- For data questions, cite specific numbers with ZaloPay priority labels (CRITICAL / ALERT / WATCH / STABLE)
- If something truly cannot be answered from any available source, say so clearly

Context available:
EXISTING REPORT (excerpt — empty if no report generated yet):
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
