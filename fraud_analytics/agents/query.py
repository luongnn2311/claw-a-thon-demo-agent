"""
Query node — runs the pandas pipeline + deterministic suggest_* analysis tools.
The LLM decides which analysis functions to call based on report_type + fraud_pillar.
"""
from __future__ import annotations
from typing import Dict, Any
from langchain_core.tools import tool
from fraud_analytics.state import FraudReportState
from fraud_analytics.tools.pipeline import run_pipeline
from fraud_analytics.tools.analysis import (
    analyze_fraud_monthly,
    analyze_fraud_weekly,
    analyze_promo_weekly,
    analyze_promo_monthly,
    analyze_coin2dd,
    analyze_appid_breakdown,
)

# ── Pipeline cache ─────────────────────────────────────────────────────────────
# Populated once by tool_run_pipeline, reused by all tool_analyze_* in the same
# query_node call. Cleared at the start of each query_node invocation.
_pipeline_cache: Dict[str, Any] = {}


def _get_pipeline_result() -> Dict[str, Any]:
    """Return cached pipeline result, running with no date filter if not yet cached."""
    if not _pipeline_cache:
        _pipeline_cache.update(run_pipeline())
    return _pipeline_cache


# ── LangChain tools wrapping the pipeline + analysis functions ────────────────

@tool
def tool_run_pipeline(start_date: str = "", end_date: str = "") -> dict:
    """Run the ZaloPay pandas pipeline from raw CSVs.
    Returns the 5 output tables as JSON records.
    Call this first before any analysis.
    Args: start_date and end_date in YYYY-MM-DD format (optional)."""
    result = run_pipeline(start_date or None, end_date or None)
    _pipeline_cache.clear()
    _pipeline_cache.update(result)
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
    records = _get_pipeline_result().get("fraud_monthly_loss", [])
    return analyze_fraud_monthly(records)

@tool
def tool_analyze_fraud_weekly() -> list:
    """Run suggest_fraud_weekly analysis on the latest fraud_weekly_loss table.
    Returns prioritized suggestions for weekly trend."""
    records = _get_pipeline_result().get("fraud_weekly_loss", [])
    return analyze_fraud_weekly(records)

@tool
def tool_analyze_promo_weekly() -> list:
    """Run suggest_promo_weekly analysis on the latest promo_weekly_abuse table.
    Returns prioritized suggestions for promo abuse trend."""
    records = _get_pipeline_result().get("promo_weekly_abuse", [])
    return analyze_promo_weekly(records)

@tool
def tool_analyze_promo_monthly() -> list:
    """Run suggest_promo_monthly analysis: aggregates all weeks in the period to compute
    monthly-level %abuse. Use for monthly reports or when monthly promo context is needed.
    Returns prioritized suggestions using monthly thresholds (normal 1.8–3.5%, alert >5%)."""
    records = _get_pipeline_result().get("promo_weekly_abuse", [])
    return analyze_promo_monthly(records)

@tool
def tool_analyze_coin2dd() -> list:
    """Run suggest_coin2dd analysis on the latest coin2dd_monthly table.
    Returns prioritized suggestions for Coin2DD abuse."""
    records = _get_pipeline_result().get("coin2dd_monthly", [])
    return analyze_coin2dd(records)

@tool
def tool_analyze_appid_breakdown() -> list:
    """Run suggest_appid_breakdown analysis on the latest appid_fraud_breakdown table.
    Returns prioritized suggestions for appID concentration and trends."""
    records = _get_pipeline_result().get("appid_fraud_breakdown", [])
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

# Direct dispatch: maps (pillar, report_type) → analysis functions to call
# Eliminates the LLM tool-calling loop entirely for known pillars.
_DISPATCH: Dict[str, Dict[str, list]] = {
    "fraud_loss":      {"weekly": [tool_analyze_fraud_weekly],
                        "monthly": [tool_analyze_fraud_monthly],
                        "adhoc":  [tool_analyze_fraud_weekly, tool_analyze_fraud_monthly]},
    "promo_abuse":     {"weekly": [tool_analyze_promo_weekly],
                        "monthly": [tool_analyze_promo_monthly],
                        "adhoc":  [tool_analyze_promo_weekly, tool_analyze_promo_monthly]},
    "coin2dd":         {"*": [tool_analyze_coin2dd]},
    "appid_breakdown": {"*": [tool_analyze_appid_breakdown]},
    "general":         {"*": [tool_analyze_fraud_weekly, tool_analyze_fraud_monthly,
                               tool_analyze_promo_weekly, tool_analyze_promo_monthly,
                               tool_analyze_coin2dd, tool_analyze_appid_breakdown]},
}


def _get_analysis_fns(pillar: str, report_type: str) -> list:
    pillar_map = _DISPATCH.get(pillar, _DISPATCH["general"])
    return pillar_map.get(report_type) or pillar_map.get("*") or list(pillar_map.values())[0]


def query_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()

    _pipeline_cache.clear()

    dr          = state.get("date_range", {})
    start_date  = dr.get("start", "")
    end_date    = dr.get("end", "")
    pillar      = state.get("fraud_pillar", "general")
    report_type = state.get("report_type", "weekly")

    existing_results: Dict[str, Any]  = dict(state.get("query_results") or {})
    existing_analysis: Dict[str, Any] = dict(state.get("analysis_results") or {})

    new_query_results: Dict[str, Any]  = {}
    new_analysis_results: Dict[str, Any] = {}

    # Step 1 — run pipeline once (populates _pipeline_cache)
    if "tool_run_pipeline" not in existing_results:
        try:
            pipeline_summary = tool_run_pipeline.invoke({"start_date": start_date, "end_date": end_date})
            new_query_results["tool_run_pipeline"] = pipeline_summary
            for tbl in ["fraud_monthly_loss", "fraud_weekly_loss",
                        "promo_weekly_abuse", "coin2dd_monthly", "appid_fraud_breakdown"]:
                if tbl in _pipeline_cache:
                    new_query_results[tbl] = _pipeline_cache[tbl]
        except Exception as exc:
            logging.getLogger(__name__).error("pipeline error: %s", exc)

    # Step 2 — run analysis functions directly based on pillar (no LLM needed)
    for fn in _get_analysis_fns(pillar, report_type):
        if fn.name in existing_analysis:
            continue
        try:
            new_analysis_results[fn.name] = fn.invoke({})
        except Exception as exc:
            logging.getLogger(__name__).error("analysis error %s: %s", fn.name, exc)

    merged_query    = {**existing_results, **new_query_results}
    merged_analysis = {**existing_analysis, **new_analysis_results}

    # Update date_range to reflect the actual last available period in the data,
    # not the orchestrator's calendar guess (which may exceed available data).
    actual_dr = dict(dr)
    if report_type == "weekly":
        tbl = "promo_weekly_abuse" if pillar == "promo_abuse" else "fraud_weekly_loss"
        rows = _pipeline_cache.get(tbl, [])
        if rows:
            last = rows[-1]
            actual_dr = {
                "start": last.get("week_start", dr.get("start", "")),
                "end":   last.get("week_end",   dr.get("end",   "")),
            }
    elif report_type == "monthly":
        tbl = "coin2dd_monthly" if pillar == "coin2dd" else "fraud_monthly_loss"
        rows = _pipeline_cache.get(tbl, [])
        if rows:
            last = rows[-1]
            actual_dr = {
                "start": last.get("period_start", dr.get("start", "")),
                "end":   last.get("period_end",   dr.get("end",   "")),
            }

    logging.getLogger(__name__).info("TIMING query_node %.1fs | actual_dr=%s", time.time() - _t0, actual_dr)
    return {
        "query_results":    merged_query,
        "analysis_results": merged_analysis,
        "date_range":       actual_dr,
        "messages": state.get("messages", []) + [{
            "role": "query",
            "content": (
                f"Pipeline: {list(new_query_results.keys())} | "
                f"Analysis: {list(new_analysis_results.keys())} | "
                f"date_range: {actual_dr}"
            ),
        }],
    }
