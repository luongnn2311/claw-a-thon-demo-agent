from __future__ import annotations
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke
from fraud_analytics.schemas.models import OrchestratorOutput

_SYSTEM = """You are the Main Orchestrator Agent for a Fraud Analytics System.

Your responsibilities:
1. Parse the user's fraud investigation or reporting request
2. Determine the appropriate report type: weekly, monthly, or adhoc
3. Identify the primary fraud pillar to investigate
4. Extract or infer specific date ranges from context

Fraud pillars:
  - merchant_abuse    : Merchant collusion, fake merchants, merchant volume manipulation
  - discount_abuse    : Promotional code abuse, discount stacking, coupon fraud
  - volume_risk       : Sudden transaction volume spikes, synthetic transactions, bot activity
  - user_abuse        : High-frequency users, account farming, identity abuse
  - payment_risk      : Failed transaction patterns, card testing, BIN attacks
  - general           : Cross-pillar or unspecified fraud analysis

Date inference rules:
  - "last week" → the 7 days ending yesterday
  - "this week" → Monday to today
  - "last month" → the full previous calendar month
  - "past 30 days" → today minus 30 days to today
  - If no date mentioned → default to last 7 days

Today's date: {today}

Respond with structured output only."""


def orchestrator_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)

    today = datetime.now().strftime("%Y-%m-%d")
    system_msg = SystemMessage(content=_SYSTEM.format(today=today))
    human_msg = HumanMessage(content=state["user_request"])

    result: OrchestratorOutput = structured_invoke(llm, [system_msg, human_msg], OrchestratorOutput)

    return {
        "report_type": result.report_type,
        "date_range": {"start": result.date_range.start, "end": result.date_range.end},
        "fraud_pillar": result.fraud_pillar,
        "retry_count": 0,
        "messages": state.get("messages", [])
        + [
            {
                "role": "orchestrator",
                "content": (
                    f"type={result.report_type} pillar={result.fraud_pillar} "
                    f"range={result.date_range.start}→{result.date_range.end}"
                ),
            }
        ],
    }
