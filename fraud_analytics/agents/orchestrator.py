from __future__ import annotations
import time
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke
from fraud_analytics.schemas.models import OrchestratorOutput

_SYSTEM = """You are the Main Orchestrator Agent for a ZaloPay Fraud Analytics System.

Your responsibilities:
1. Parse the user's fraud investigation or reporting request
2. Determine the appropriate report type: weekly, monthly, or adhoc
3. Identify the primary fraud pillar to investigate
4. Extract or infer specific date ranges from context
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

Date inference rules — ALWAYS include one comparison baseline period so WoW/MoM diffs are available:

  Weekly report:
    - "last week" or no date → start = Monday of the week BEFORE last week, end = last Sunday
      Example: if today is 2026-06-15 (Mon), last complete week is Jun 8–14,
               set start = 2026-06-01 (Mon 2 weeks ago), end = 2026-06-14
    - "this week" → start = Monday of last week, end = today
    - Specific week → start = Monday of the PREVIOUS week, end = Sunday of the requested week

  Monthly report:
    - "last month" or no date → start = 1st of the month BEFORE last month, end = last day of last month
      Example: if today is 2026-06-15, last complete month is May,
               set start = 2026-04-01, end = 2026-05-31
    - "this month" → start = 1st of last month, end = today
    - Specific month → start = 1st of the PREVIOUS month, end = last day of the requested month

  General rule: always include ONE extra period before the requested window so diffs are computable.

Today's date: {today}

Respond with structured output only."""


def orchestrator_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)

    today = datetime.now().strftime("%Y-%m-%d")
    system_msg = SystemMessage(content=_SYSTEM.format(today=today))
    human_msg = HumanMessage(content=state["user_request"])

    result: OrchestratorOutput = structured_invoke(llm, [system_msg, human_msg], OrchestratorOutput)

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

    return {
        "report_type":   result.report_type,
        "date_range":    {"start": result.date_range.start, "end": result.date_range.end},
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
                    f"range={result.date_range.start}→{result.date_range.end}"
                ),
            }
        ],
    }
