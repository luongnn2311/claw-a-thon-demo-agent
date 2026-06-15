"""
Query node — runs the pandas pipeline + deterministic suggest_* analysis tools.
The LLM decides which analysis functions to call based on report_type + fraud_pillar.
"""
from __future__ import annotations
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm
from fraud_analytics.tools.pipeline import run_pipeline
from fraud_analytics.tools.analysis import (
    analyze_fraud_monthly,
    analyze_fraud_weekly,
    analyze_promo_weekly,
    analyze_promo_monthly,
    analyze_coin2dd,
    analyze_appid_breakdown,
)

# ── LangChain tools wrapping the pipeline + analysis functions ────────────────

@tool
def tool_run_pipeline(start_date: str = "", end_date: str = "") -> dict:
    """Run the ZaloPay pandas pipeline from raw CSVs.
    Returns the 5 output tables as JSON records.
    Call this first before any analysis.
    Args: start_date and end_date in YYYY-MM-DD format (optional)."""
    result = run_pipeline(start_date or None, end_date or None)
    # Return only table summaries to keep token usage manageable
    summary = {"success": result.get("success"), "tables_computed": result.get("tables_computed", [])}
    for table in ["fraud_monthly_loss", "fraud_weekly_loss", "promo_weekly_abuse",
                  "coin2dd_monthly", "appid_fraud_breakdown"]:
        if table in result:
            summary[f"{table}_rows"] = len(result[table])
            summary[f"{table}_sample"] = result[table][-2:] if result[table] else []
    return summary

@tool
def tool_analyze_fraud_monthly() -> list:
    """Run suggest_fraud_monthly analysis on the latest fraud_monthly_loss table.
    Returns prioritized suggestions (CRITICAL/ALERT/WATCH/STABLE/CONFIRM/INVESTIGATE)."""
    result = run_pipeline()
    records = result.get("fraud_monthly_loss", [])
    return analyze_fraud_monthly(records)

@tool
def tool_analyze_fraud_weekly() -> list:
    """Run suggest_fraud_weekly analysis on the latest fraud_weekly_loss table.
    Returns prioritized suggestions for weekly trend."""
    result = run_pipeline()
    records = result.get("fraud_weekly_loss", [])
    return analyze_fraud_weekly(records)

@tool
def tool_analyze_promo_weekly() -> list:
    """Run suggest_promo_weekly analysis on the latest promo_weekly_abuse table.
    Returns prioritized suggestions for promo abuse trend."""
    result = run_pipeline()
    records = result.get("promo_weekly_abuse", [])
    return analyze_promo_weekly(records)

@tool
def tool_analyze_promo_monthly() -> list:
    """Run suggest_promo_monthly analysis: aggregates all weeks in the period to compute
    monthly-level %abuse. Use for monthly reports or when monthly promo context is needed.
    Returns prioritized suggestions using monthly thresholds (normal 1.8–3.5%, alert >5%)."""
    result = run_pipeline()
    records = result.get("promo_weekly_abuse", [])
    return analyze_promo_monthly(records)

@tool
def tool_analyze_coin2dd() -> list:
    """Run suggest_coin2dd analysis on the latest coin2dd_monthly table.
    Returns prioritized suggestions for Coin2DD abuse."""
    result = run_pipeline()
    records = result.get("coin2dd_monthly", [])
    return analyze_coin2dd(records)

@tool
def tool_analyze_appid_breakdown() -> list:
    """Run suggest_appid_breakdown analysis on the latest appid_fraud_breakdown table.
    Returns prioritized suggestions for appID concentration and trends."""
    result = run_pipeline()
    records = result.get("appid_fraud_breakdown", [])
    return analyze_appid_breakdown(records)


ALL_TOOLS = [
    tool_run_pipeline,
    tool_analyze_fraud_monthly,
    tool_analyze_fraud_weekly,
    tool_analyze_promo_weekly,
    tool_analyze_promo_monthly,
    tool_analyze_coin2dd,
    tool_analyze_appid_breakdown,
]

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}

_SYSTEM = """You are the Data Query Agent for a ZaloPay Fraud Analytics System.

Your job is to call the right tools to gather and analyze data for the fraud report.

TOOL CALLING STRATEGY:
1. ALWAYS call tool_run_pipeline first to load all 5 tables.
2. Then call the relevant analysis tool(s) based on fraud_pillar and tables_to_use:
   - fraud_loss / weekly      → tool_analyze_fraud_weekly
   - fraud_loss / monthly     → tool_analyze_fraud_monthly
   - promo_abuse / weekly     → tool_analyze_promo_weekly
   - promo_abuse / monthly    → tool_analyze_promo_monthly (aggregates weeks → monthly %abuse)
   - coin2dd                  → tool_analyze_coin2dd
   - appid_breakdown          → tool_analyze_appid_breakdown
   - general                  → call ALL analysis tools

Context:
  report_type   : {report_type}
  fraud_pillar  : {fraud_pillar}
  tables_to_use : {tables_to_use}
  date_range    : {start_date} to {end_date}
  retry_count   : {retry_count}
  already_called: {already_called}

Rules:
- Do NOT call tools already in already_called — cached results are reused automatically.
- Stop after all relevant analysis tools have been called."""

_MAX_LOOP = 10


def query_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    dr = state.get("date_range", {})
    start_date = dr.get("start", "")
    end_date = dr.get("end", "")
    existing_results: Dict[str, Any] = dict(state.get("query_results") or {})
    existing_analysis: Dict[str, Any] = dict(state.get("analysis_results") or {})

    system_content = _SYSTEM.format(
        report_type=state.get("report_type", "weekly"),
        fraud_pillar=state.get("fraud_pillar", "general"),
        tables_to_use=state.get("tables_to_use", []),
        start_date=start_date,
        end_date=end_date,
        retry_count=state.get("retry_count", 0),
        already_called=list(existing_results.keys()) + list(existing_analysis.keys()),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=(
            f"Run the pipeline and analyze: {state.get('fraud_pillar', 'general')} "
            f"({state.get('report_type', 'weekly')}) from {start_date} to {end_date}."
        )),
    ]

    new_query_results: Dict[str, Any] = {}
    new_analysis_results: Dict[str, Any] = {}

    for _ in range(_MAX_LOOP):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            tool_name = tc["name"]
            all_done = {**new_query_results, **new_analysis_results,
                        **existing_results, **existing_analysis}
            if tool_name in all_done:
                messages.append(ToolMessage(
                    content=str(all_done[tool_name]), tool_call_id=tc["id"]
                ))
                continue

            tool_fn = _TOOL_MAP.get(tool_name)
            if tool_fn is None:
                result = f"Unknown tool: {tool_name}"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"

            # Route to correct bucket
            if tool_name == "tool_run_pipeline":
                new_query_results[tool_name] = result
                # Also store the full pipeline output for downstream nodes
                if isinstance(result, dict) and result.get("success"):
                    full = run_pipeline(start_date or None, end_date or None)
                    for tbl in ["fraud_monthly_loss", "fraud_weekly_loss",
                                "promo_weekly_abuse", "coin2dd_monthly", "appid_fraud_breakdown"]:
                        if tbl in full:
                            new_query_results[tbl] = full[tbl]
            else:
                new_analysis_results[tool_name] = result

            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    merged_query = {**existing_results, **new_query_results}
    merged_analysis = {**existing_analysis, **new_analysis_results}

    logging.getLogger(__name__).info("TIMING query_node %.1fs", time.time() - _t0)
    return {
        "query_results": merged_query,
        "analysis_results": merged_analysis,
        "messages": state.get("messages", []) + [{
            "role": "query",
            "content": (
                f"Pipeline: {list(new_query_results.keys())} | "
                f"Analysis: {list(new_analysis_results.keys())}"
            ),
        }],
    }
