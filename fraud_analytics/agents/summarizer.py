from __future__ import annotations
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

_SYSTEM = """You are the Data Summarization Agent for a Fraud Analytics System.

Convert raw tool output into a concise, business-readable insight (2-4 sentences).

Rules:
- Always include specific numbers and percentages from the data.
- Compare to thresholds or historical norms when information is available.
- Flag anything that appears abnormal or suspicious.
- Write for a fraud risk manager, not a data scientist.
- One summary per data source. Start directly with the insight — no preamble.

Context: {report_type} | {fraud_pillar} | {start_date} to {end_date}"""


def summarizer_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)

    query_results = state.get("query_results") or {}
    if not query_results:
        return {
            "summaries": ["No query data available — summary skipped."],
            "messages": state.get("messages", []),
        }

    dr = state.get("date_range", {})
    system_msg = SystemMessage(
        content=_SYSTEM.format(
            report_type=state.get("report_type", "weekly"),
            fraud_pillar=state.get("fraud_pillar", "general"),
            start_date=dr.get("start", "N/A"),
            end_date=dr.get("end", "N/A"),
        )
    )

    summaries: List[str] = []
    for tool_name, result in query_results.items():
        try:
            data_str = json.dumps(result, indent=2, default=str)[:2000]
        except Exception:
            data_str = str(result)[:2000]

        human_msg = HumanMessage(
            content=f"Summarize this `{tool_name}` result:\n\n{data_str}"
        )
        response = llm.invoke([system_msg, human_msg])
        label = tool_name.replace("_", " ").title()
        summaries.append(f"**{label}**: {response.content.strip()}")

    return {
        "summaries": summaries,
        "messages": state.get("messages", [])
        + [
            {
                "role": "summarizer",
                "content": f"Generated {len(summaries)} data summaries",
            }
        ],
    }
