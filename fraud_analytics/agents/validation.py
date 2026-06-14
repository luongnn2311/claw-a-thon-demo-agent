from __future__ import annotations
import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke, MAX_VALIDATION_RETRIES


# ── Output schema ─────────────────────────────────────────────────────────────

class ValidationIssue(BaseModel):
    issue: str = Field(description="What is wrong")
    severity: Literal["minor", "major", "blocking"] = Field(description="Impact on report quality")
    suggested_fix: str = Field(description="How to resolve — be specific about what to re-query or retrieve")


class ValidationOutput(BaseModel):
    validated: bool = Field(description="True if findings are ready for report generation")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence 0-1")
    issues_found: List[ValidationIssue] = Field(default_factory=list)
    next_step: Literal["report", "query", "retrieval"] = Field(
        description="Next step: report (pass), query (re-run data tools), retrieval (need more knowledge)"
    )
    validation_notes: str = Field(description="Summary of the validation assessment")


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are the Validation Agent for ZaloPay Fraud Analytics.

Your job: verify that the reasoning agent's findings accurately reflect what
the deterministic suggest_* analysis tools produced, and are ready for report generation.

━━━ ZaloPay PRIORITY LABELS (ground truth) ━━━
  CRITICAL — threshold breached, same-day action required
  ALERT    — approaching threshold, action within 24 h
  WATCH    — elevated but within range, monitor this week
  STABLE   — within normal range

━━━ ZaloPay THRESHOLD REFERENCE ━━━
  fraud_loss:      MoM total_loss > +20% → CRITICAL;  +10-20% → ALERT
  promo_abuse:     pct_abuse > 40% → CRITICAL;  30-40% → ALERT
  coin2dd:         pct_abuse > 20% → CRITICAL;  10-20% → ALERT
  appid_breakdown: MoM fraud_loss > +50% for a single appID → CRITICAL

━━━ VALIDATION CHECKLIST ━━━
1. Coverage  — every CRITICAL or ALERT item from analysis_results must appear in findings
2. Numbers   — findings must cite specific figures (%, M VND, counts) from pipeline data
3. Priority  — severity labels in findings must match analysis_results priority labels
4. Scope     — findings must stay within the declared fraud_pillar and tables_to_use
5. Period    — findings must reference the correct period (week_start/week_end for weekly,
               period_start/period_end for monthly; note is_partial for current month)
6. Actions   — each CRITICAL/ALERT finding must have a concrete ZaloPay-specific action

━━━ RETRY DECISION ━━━
  → validated=false, next_step="query"      when: CRITICAL items in analysis_results have
                                             no corresponding finding, or numbers are missing
  → validated=false, next_step="retrieval"  when: findings exist but lack ZaloPay context
                                             or recommended actions are too generic
  → validated=true,  next_step="report"     when: all CRITICAL/ALERT items covered with
                                             specific numbers and concrete actions
  → ALWAYS force validated=true, next_step="report" when retry_count >= {max_retries}

━━━ CURRENT STATE ━━━
  retry_count        : {retry_count} / {max_retries}
  fraud_pillar       : {fraud_pillar}
  tables_to_use      : {tables_to_use}
  findings_count     : {findings_count}
  analysis_items     : {analysis_items}
  critical_alerts    : {critical_alerts}
  docs_retrieved     : {docs_count}
"""


def _count_critical_alerts(analysis_results: Dict[str, Any]) -> int:
    """Count total CRITICAL + ALERT items across all analysis_results."""
    count = 0
    for items in analysis_results.values():
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    p = item.get("priority", "")
                    if p in ("CRITICAL", "ALERT"):
                        count += 1
    return count


def _total_analysis_items(analysis_results: Dict[str, Any]) -> int:
    total = 0
    for items in analysis_results.values():
        if isinstance(items, list):
            total += len(items)
    return total


def validation_node(state: FraudReportState) -> Dict[str, Any]:
    retry_count = state.get("retry_count", 0)

    # Hard circuit breaker — force pass at max retries
    if retry_count >= MAX_VALIDATION_RETRIES:
        validation_result = {
            "validated": True,
            "confidence": 0.70,
            "issues_found": [],
            "next_step": "report",
            "validation_notes": (
                f"Max retries ({MAX_VALIDATION_RETRIES}) reached — proceeding to report "
                "with available findings."
            ),
        }
        return {
            "validation_result": validation_result,
            "retry_count": retry_count,
            "messages": state.get("messages", []) + [{
                "role": "validation",
                "content": f"Max retries reached — forced pass → report",
            }],
        }

    llm = get_llm(temperature=0.1)

    findings        = state.get("findings") or []
    analysis_results= state.get("analysis_results") or {}
    query_results   = state.get("query_results") or {}
    docs            = state.get("retrieved_documents") or []
    fraud_pillar    = state.get("fraud_pillar", "general")
    tables_to_use   = state.get("tables_to_use") or []

    critical_alerts  = _count_critical_alerts(analysis_results)
    analysis_items   = _total_analysis_items(analysis_results)

    # Fast-pass: no analysis results at all (pipeline produced nothing) → skip to report
    if analysis_items == 0 and len(findings) == 0:
        validation_result = {
            "validated": True,
            "confidence": 0.55,
            "issues_found": [],
            "next_step": "report",
            "validation_notes": "No analysis results or findings — pipeline may have no data for this period.",
        }
        return {
            "validation_result": validation_result,
            "retry_count": retry_count,
            "messages": state.get("messages", []) + [{
                "role": "validation",
                "content": "No data — fast-pass → report",
            }],
        }

    # Build context for LLM
    try:
        findings_json  = json.dumps(findings,         indent=2, default=str)[:3000]
        analysis_json  = json.dumps(analysis_results, indent=2, default=str)[:2000]
    except Exception:
        findings_json  = str(findings)[:3000]
        analysis_json  = str(analysis_results)[:2000]

    system_content = _SYSTEM.format(
        max_retries   = MAX_VALIDATION_RETRIES,
        retry_count   = retry_count,
        fraud_pillar  = fraud_pillar,
        tables_to_use = tables_to_use,
        findings_count= len(findings),
        analysis_items= analysis_items,
        critical_alerts=critical_alerts,
        docs_count    = len(docs),
    )

    human_content = (
        f"ANALYSIS RESULTS (from suggest_* tools — ground truth):\n{analysis_json}\n\n"
        f"FINDINGS (from reasoning agent — {len(findings)} items):\n{findings_json}\n\n"
        f"Tables used: {tables_to_use}\n"
        f"Pipeline data keys: {list(query_results.keys())}\n"
        f"Knowledge docs retrieved: {len(docs)}\n\n"
        "Validate and decide: are these findings ready for report generation?"
    )

    result: ValidationOutput = structured_invoke(
        llm,
        [SystemMessage(content=system_content), HumanMessage(content=human_content)],
        ValidationOutput,
    )

    if result is None:
        return {
            "validation_result": {
                "validated": True,
                "confidence": 0.60,
                "issues_found": [],
                "next_step": "report",
                "validation_notes": "Validation parsing failed — proceeding to report.",
            },
            "retry_count": MAX_VALIDATION_RETRIES,
            "messages": state.get("messages", []) + [{
                "role": "validation", "content": "PARSE FAILED — forced pass → report",
            }],
        }

    new_retry = retry_count + (0 if result.validated else 1)

    return {
        "validation_result": result.model_dump(),
        "retry_count": new_retry,
        "messages": state.get("messages", []) + [{
            "role": "validation",
            "content": (
                f"{'PASSED' if result.validated else 'FAILED'} "
                f"confidence={result.confidence:.2f} "
                f"critical_alerts_in_analysis={critical_alerts} "
                f"issues={len(result.issues_found)} "
                f"next={result.next_step}"
            ),
        }],
    }
