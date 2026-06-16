from __future__ import annotations
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke
from fraud_analytics.schemas.models import FraudAnalysisOutput

_SYSTEM = """You are a Senior ZaloPay Risk Analyst applying fraud investigation decision trees.

Your job is to synthesize the narrative summaries and domain knowledge to produce
structured findings with specific investigation priorities.

DECISION TREE — apply in order:
Fraud Loss:
  IF total_loss MoM > +20% → check international first, then domestic_direct, then napas
  IF international up → check appIDs 149, 3762, 9999 → rule or BIN limit
  IF domestic_direct up → check appID 454 VCB sub-segment → behavioral controls
  IF domestic_napas up → check BIN concentration + SME → block BIN/merchant
  IF 45% undrained → report only the drained 55% as actual loss

Promo Abuse:
  IF %abuse > 4% → CRITICAL: check campaign splitting; deploy challenge; Coin2DD path
  IF %abuse < 1.5% AND detection degraded → CAUTION: not improvement, visibility loss
  IF Coin2DD > 7% → CRITICAL: SME earn path contributing; apply SME earn cap
  IF detection drop (FAD + BAD_V2 both declining) → reactivate legacy rules; escalate DS

Cross-domain:
  Declining metrics ≠ improvement if detection is also declining
  Single-week anomaly → investigate before reporting as trend
  Control effect → expect ~50% drop within same deployment week

SEVERITY CALIBRATION:
  CRITICAL = requires immediate action (same day)
  HIGH = action within 24h
  MEDIUM = action within this week
  LOW = monitor / BAU

Domain knowledge context:
{knowledge}

Narrative summaries from analysis:
{summaries}

Raw analysis findings (structured suggest_* output):
{analysis_findings}

Scope: {report_type} | {fraud_pillar} | {start_date} to {end_date}"""


def reasoning_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()
    llm = get_llm(temperature=0.3)

    docs = state.get("retrieved_documents") or []
    summaries = state.get("summaries") or []
    analysis_results = state.get("analysis_results") or {}
    dr = state.get("date_range", {})

    knowledge = "\n\n".join(
        f"[{d.get('metadata', {}).get('source', '?')}] {d['content']}"
        for d in docs[:6]
    )

    try:
        analysis_findings = json.dumps(analysis_results, indent=2, default=str)[:3000]
    except Exception:
        analysis_findings = str(analysis_results)[:3000]

    system_content = _SYSTEM.format(
        knowledge=knowledge or "No additional domain docs retrieved.",
        summaries="\n\n".join(summaries) or "No summaries available.",
        analysis_findings=analysis_findings,
        report_type=state.get("report_type", "weekly"),
        fraud_pillar=state.get("fraud_pillar", "general"),
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
    )

    validation_feedback = state.get("validation_feedback") or ""
    human_content = (
        "Apply the decision trees above to the analysis findings. "
        "Produce findings with SPECIFIC numbers and concrete recommended_actions. "
        "Cover all tables with CRITICAL or ALERT priority findings first."
    )
    if validation_feedback:
        human_content += f"\n\n{validation_feedback}"

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    result: FraudAnalysisOutput = structured_invoke(llm, messages, FraudAnalysisOutput)

    findings = [f.model_dump() for f in result.findings]

    logging.getLogger(__name__).info("TIMING reasoning_node %.1fs", time.time() - _t0)
    return {
        "findings": findings,
        "validation_feedback": "",  # consumed — clear so it doesn't bleed into next cycle
        "messages": state.get("messages", []) + [{
            "role": "reasoning",
            "content": (
                f"Identified {len(findings)} findings. "
                f"Overall risk: {result.overall_risk_level}"
            ),
        }],
    }
