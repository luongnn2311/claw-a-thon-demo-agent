from __future__ import annotations
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke
from fraud_analytics.schemas.models import FraudAnalysisOutput

_SYSTEM = """You are a Senior Fraud Analyst with 10+ years of experience in digital payments fraud.

Analyze the data summaries and retrieved policy documents to produce structured fraud findings.

Fraud dimensions to evaluate:
  1. Volume Risk        — transaction spikes (>200% vs baseline), Z-score anomalies
  2. Discount Abuse     — discount_ratio > 40% per merchant, coordinated code usage
  3. Merchant Abuse     — top-5 merchant concentration >60%, Z-score outlier merchants
  4. User Abuse         — high-frequency users exceeding thresholds, new account velocity
  5. Payment Risk       — failure rate >25% on any solution, potential card testing
  6. Emerging Risks     — patterns not matching known categories

For every finding:
  - Include SPECIFIC numbers from the evidence (do not generalize)
  - Calibrate severity: critical(requires immediate action), high(action within 24h),
    medium(action within week), low(monitor)
  - Set confidence based on data completeness: 0.9+ = strong evidence, 0.7-0.9 = moderate,
    <0.7 = indicative only
  - Give a concrete recommended_action

Policy context:
{knowledge}

Data summaries:
{summaries}

Supporting query data (truncated):
{query_data}

Analysis scope: {report_type} | {fraud_pillar} | {start_date} to {end_date}"""


def reasoning_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.3)

    docs = state.get("retrieved_documents") or []
    summaries = state.get("summaries") or []
    query_results = state.get("query_results") or {}
    dr = state.get("date_range", {})

    knowledge = "\n\n".join(
        f"[{d.get('metadata', {}).get('source', '?')}] {d['content']}"
        for d in docs[:6]
    )

    try:
        query_data = json.dumps(
            {k: v for k, v in list(query_results.items())[:4]},
            indent=2,
            default=str,
        )[:3500]
    except Exception:
        query_data = str(query_results)[:3500]

    system_content = _SYSTEM.format(
        knowledge=knowledge or "No policy documents retrieved.",
        summaries="\n".join(summaries) or "No summaries available.",
        query_data=query_data,
        report_type=state.get("report_type", "weekly"),
        fraud_pillar=state.get("fraud_pillar", "general"),
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=(
                "Analyze all available evidence and generate comprehensive fraud findings. "
                "Be specific, cite numbers, and cover all relevant fraud dimensions."
            )
        ),
    ]

    result: FraudAnalysisOutput = structured_invoke(llm, messages, FraudAnalysisOutput)

    findings = [f.model_dump() for f in result.findings]

    return {
        "findings": findings,
        "messages": state.get("messages", [])
        + [
            {
                "role": "reasoning",
                "content": (
                    f"Identified {len(findings)} findings. "
                    f"Overall risk: {result.overall_risk_level}"
                ),
            }
        ],
    }
