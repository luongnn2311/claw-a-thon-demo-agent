from __future__ import annotations
import time
from typing import Dict, Any, List
from datetime import datetime, date, timedelta
from calendar import monthrange
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke
from fraud_analytics.schemas.models import OrchestratorOutput

_SYSTEM = """You are the Main Orchestrator Agent for a ZaloPay Fraud Analytics System.

Your responsibilities:
1. Parse the user's fraud investigation or reporting request
2. Determine the appropriate report type: weekly, monthly, or adhoc
3. Identify the primary fraud pillar to investigate
4. Set date_range to the REPORTING period only (the system adds the baseline automatically)
5. Determine which of the 5 output tables are relevant

Fraud pillars:
  - fraud_loss      : Fraud monthly/weekly loss by segment (domestic_direct, international, napas, wallet)
  - promo_abuse     : Promotion abuse weekly metrics (%abuse, totalAbuse, abuser users)
  - coin2dd         : Coin-to-direct-debit monthly abuse metrics
  - appid_breakdown : Per-appID fraud concentration and trend
  - general         : Multi-pillar analysis (use all 5 tables)

Output tables mapping:
  - fraud_loss      → fraud_monthly_loss + fraud_weekly_loss
  - promo_abuse     → promo_weekly_abuse
  - coin2dd         → coin2dd_monthly
  - appid_breakdown → appid_fraud_breakdown
  - general         → all 5 tables

Date inference rules (REPORTING period only — baseline is added automatically):
  - "last week" or no date for weekly  → last complete Mon–Sun week
  - "this week"                        → Monday of current week to today
  - "last month" or no date for monthly→ previous full calendar month
  - "this month"                       → 1st of current month to today
  - Specific date mentioned            → use that date range exactly

Today's date: {today}

Respond with structured output only."""


def _default_weekly_range(today: date) -> tuple[str, str]:
    """Last complete Mon–Sun week. Returns (start, end) as YYYY-MM-DD strings."""
    # dayofweek: Mon=0 … Sun=6
    days_since_sunday = (today.weekday() + 1) % 7  # days since last Sunday
    last_sunday = today - timedelta(days=days_since_sunday)
    last_monday = last_sunday - timedelta(days=6)
    return last_monday.strftime("%Y-%m-%d"), last_sunday.strftime("%Y-%m-%d")


def _default_monthly_range(today: date) -> tuple[str, str]:
    """Last complete calendar month. Returns (start, end) as YYYY-MM-DD strings."""
    first_this_month = today.replace(day=1)
    last_month_end   = first_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    return last_month_start.strftime("%Y-%m-%d"), last_month_end.strftime("%Y-%m-%d")


def _add_baseline(report_type: str, start: str, end: str) -> tuple[str, str]:
    """Extend start backward by one period so WoW/MoM diffs are computable."""
    try:
        s = datetime.strptime(start, "%Y-%m-%d").date()
        if report_type == "weekly":
            new_start = s - timedelta(days=7)
        else:  # monthly — go back to the 1st of the previous month
            first_of_start = s.replace(day=1)
            prev_month_end = first_of_start - timedelta(days=1)
            new_start = prev_month_end.replace(day=1)
        return new_start.strftime("%Y-%m-%d"), end
    except Exception:
        return start, end


def orchestrator_node(state: FraudReportState) -> Dict[str, Any]:
    import logging
    _t0 = time.time()
    llm = get_llm(temperature=0.0)

    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    system_msg = SystemMessage(content=_SYSTEM.format(today=today_str))
    human_msg = HumanMessage(content=state["user_request"])

    result: OrchestratorOutput = structured_invoke(llm, [system_msg, human_msg], OrchestratorOutput)

    # Override date range with deterministic calculation for standard report types
    # (LLMs are unreliable at computing "Monday 2 weeks ago" correctly)
    start, end = result.date_range.start, result.date_range.end
    if result.report_type == "weekly":
        start, end = _default_weekly_range(today)
    elif result.report_type == "monthly":
        start, end = _default_monthly_range(today)

    # Extend start backward by one period to include the WoW/MoM baseline
    start, end = _add_baseline(result.report_type, start, end)

    # Determine which tables to use based on pillar
    pillar = result.fraud_pillar
    table_map: Dict[str, List[str]] = {
        "fraud_loss":      ["fraud_monthly_loss", "fraud_weekly_loss"],
        "promo_abuse":     ["promo_weekly_abuse"],
        "coin2dd":         ["coin2dd_monthly"],
        "appid_breakdown": ["appid_fraud_breakdown"],
        "general":         ["fraud_monthly_loss", "fraud_weekly_loss",
                            "promo_weekly_abuse", "coin2dd_monthly", "appid_fraud_breakdown"],
    }
    tables_to_use = table_map.get(pillar, table_map["general"])

    logging.getLogger(__name__).info("TIMING orchestrator_node %.1fs", time.time() - _t0)
    return {
        "report_type":   result.report_type,
        "date_range":    {"start": start, "end": end},
        "fraud_pillar":  pillar,
        "tables_to_use": tables_to_use,
        "retry_count":   0,
        "total_node_visits": 1,
        "pipeline_start_time": time.time(),
        "messages": state.get("messages", []) + [
            {
                "role": "orchestrator",
                "content": (
                    f"type={result.report_type} pillar={pillar} "
                    f"tables={tables_to_use} "
                    f"range={start}→{end}"
                ),
            }
        ],
    }
