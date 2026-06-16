"""
Follow-up QA node — answers user questions about an existing report
without re-running the full pipeline.

Uses: existing report + findings + analysis_results + knowledge retrieval
+ optional targeted analysis tools (no pipeline re-run).
"""
from __future__ import annotations
import json
import pandas as pd
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
    analyze_promo_weekly, analyze_promo_monthly,
    analyze_coin2dd, analyze_appid_breakdown,
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
    """Get flags/alerts for the LATEST WEEK ONLY (compares most recent week vs prior week).
    Use this ONLY when the user asks about the current week's status or latest WoW change.
    Do NOT use for multi-week trends or 'last N weeks' requests — use get_raw_table instead."""
    r = run_pipeline()
    return analyze_fraud_weekly(r.get("fraud_weekly_loss", []))

@tool
def get_promo_detail() -> list:
    """Get detailed promo abuse analysis for the latest weeks."""
    r = run_pipeline()
    return analyze_promo_weekly(r.get("promo_weekly_abuse", []))

@tool
def get_promo_monthly_detail() -> list:
    """Get monthly-level promo abuse analysis (aggregates all weeks in the period).
    Uses monthly thresholds: normal 1.8–3.5%, alert >5%."""
    r = run_pipeline()
    return analyze_promo_monthly(r.get("promo_weekly_abuse", []))

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
def get_raw_table(table_name: str) -> str:
    """Get historical records from a summary output table as a markdown table (up to 20 rows).
    Use this for: multi-week trends, 'last N weeks/months', historical comparisons, or when
    the user wants raw data to format themselves.
    table_name must be one of: fraud_monthly_loss, fraud_weekly_loss,
    promo_weekly_abuse, coin2dd_monthly, appid_fraud_breakdown."""
    r = run_pipeline()
    data = r.get(table_name, [])
    if not data:
        return f"No data found for table `{table_name}`."
    df = pd.DataFrame(data)
    total = len(df)
    shown = min(total, 20)
    tail = df.tail(shown).reset_index(drop=True)
    cols = list(tail.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in tail.iterrows()]
    md = "\n".join([header, sep] + rows)
    note = f"\n\n*Showing last {shown} of {total} rows.*" if total > shown else f"\n\n*{total} rows total.*"
    return md + note


@tool
def query_input_files(question: str) -> str:
    """Generate and run pandas code to answer a question about the raw input CSV files.

    Use this when the user asks about:
    - Individual transactions, specific users, campaigns, or promotions
    - Raw source data (not the summary output tables)
    - Custom aggregations or filters on input files
    - Anything in: promo_transactions_with_abuse_flag, raw_clean_tpe_log,
      raw_post_mortem_account_risk(_gateway), raw_promo_abuse_dim_user_v2,
      raw_dim_user_promo_profile, raw_promotion_transaction_history

    Results are automatically capped at 20 rows and returned as a markdown table.
    """
    from fraud_analytics.tools.code_executor import execute_pandas_code, build_schema_context

    schema = build_schema_context()
    code_gen_prompt = (
        "You are a pandas code generator for ZaloPay fraud analytics.\n\n"
        "The following DataFrames are already loaded in the execution namespace.\n"
        "Do NOT use pd.read_csv — the data is already available:\n\n"
        f"{schema}\n\n"
        "Rules:\n"
        "1. Use ONLY pandas (available as `pd`) and the pre-loaded DataFrames above.\n"
        "2. Do NOT write any import statements.\n"
        "3. Assign your final answer to a variable named `result` (must be a DataFrame or scalar).\n"
        "4. If result is a DataFrame it will be capped at 20 rows automatically.\n"
        "5. Output ONLY the Python code — no explanations, no markdown fences."
    )
    llm = get_llm(temperature=0.0)
    response = llm.invoke([
        SystemMessage(content=code_gen_prompt),
        HumanMessage(content=question),
    ])
    return execute_pandas_code(response.content.strip())

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
    get_promo_monthly_detail,
    get_coin2dd_detail,
    get_appid_detail,
    get_raw_table,
    query_input_files,
    search_fintech_web,
]
_TOOL_MAP = {t.name: t for t in _FOLLOWUP_TOOLS}

_SYSTEM = """You are a ZaloPay Fraud Analytics Assistant — a domain expert on ZaloPay fraud, risk, and promo abuse.

CONVERSATION CONTEXT RULE (most important):
The conversation history is included as actual messages BEFORE the user's current question.
Always read those prior turns to resolve any ambiguous references ("it", "that", "those",
"why is it high", "tell me more", "what about last week") before answering.
If the current question is a follow-up, anchor your answer explicitly to what was said before —
e.g. "Regarding the VNPAY loss we just discussed…" — so the reply reads as a coherent continuation.

You help with THREE types of questions:
  A. General questions — domain concepts, ZaloPay terminology, fraud patterns, thresholds,
     team ownership, "what is X", "explain Y", "how does Z work"
  B. Summary data questions — specific numbers from ZaloPay summary output tables or an existing report.
     Use get_raw_table or the get_*_detail analysis tools.
  C. Raw input data questions — queries about individual transactions, users, campaigns,
     or anything in the source CSV files. Use query_input_files.

TOOL USAGE RULES — follow strictly:
1. "last N weeks" / "last N months" / multi-period requests
   → ALWAYS use get_raw_table — NEVER use get_fraud_weekly_detail or get_fraud_monthly_detail
   → table_name: fraud_weekly_loss (fraud weekly), promo_weekly_abuse (promo), fraud_monthly_loss (fraud monthly), coin2dd_monthly (Coin2DD)
   → After getting the table, present ALL returned rows within the requested period; do not truncate to the latest row only
2. get_raw_table / get_*_detail tools
   → call when the user asks for specific metrics/figures from the summary output tables
   → the tool returns a pre-formatted markdown table — pass it through to the user AS-IS, do not reformat
3. query_input_files
   → call when the user asks about raw/source data: individual transactions, specific users,
     campaign details, or custom aggregations on input files
   → the tool generates and runs pandas code and returns a markdown table — pass it through AS-IS
4. search_fintech_web
   → call when:
      a. The user explicitly asks to search / look online, OR
      b. The question is about an industry concept not covered by local knowledge
   → do NOT call if the local knowledge already fully answers the question

ANSWER RULES:
- FORMAT FIRST: If the user specifies an output format or structure in their question, follow it EXACTLY.
  Do not impose the standard report template. The user's requested format overrides everything.
- Open with a one-sentence context anchor when the question is a follow-up (reference the prior topic)
- When a tool returns a markdown table, present it directly — do not paraphrase or summarize the rows
- For concept/domain questions, answer from domain knowledge — no data tools needed
- For data questions, cite specific numbers with ZaloPay priority labels (CRITICAL / ALERT / WATCH / STABLE)
- If data covers fewer periods than requested (e.g., user asks for 6 weeks but only 2 available), say so clearly
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

Recent conversation (from graph state — use to resolve references like "it", "that", "why"):
{history}
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

    # Last 4 turns from graph state (excludes current question which is user_request)
    history = list(state.get("conversation_history") or [])
    recent = history[-5:-1] if len(history) > 1 else []
    history_text = "\n".join(
        f"  {m['role'].upper()}: {m['content'][:300]}" for m in recent
    ) or "  (none)"

    system_content = _SYSTEM.format(
        report_excerpt=report_excerpt,
        findings=findings_str,
        analysis=analysis_str,
        knowledge=knowledge or "No additional knowledge retrieved.",
        report_type=state.get("report_type", "N/A"),
        fraud_pillar=state.get("fraud_pillar", "N/A"),
        date_range=f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}",
        history=history_text,
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

    history = list(history)
    history.append({"role": "assistant", "content": answer})

    return {
        "agent_message": answer,
        "conversation_history": history,
        "messages": state.get("messages", []) + [{
            "role": "followup",
            "content": f"Answered follow-up: {question[:80]}",
        }],
    }
