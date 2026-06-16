from __future__ import annotations
import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

_REPORT_SYSTEM = """You are generating a complete ZaloPay Fraud Analytics Report.
Audience: VP Risk + Fraud Operations team.

Structure the report EXACTLY as follows (no extra sections):

════════════════════════════════════════════════════════════════════════
  EXECUTIVE SUMMARY
════════════════════════════════════════════════════════════════════════

[Lead with the single most critical finding + metric]
[4-6 bullets with specific numbers: loss M VND, % change, threshold comparison]
[Exactly 3 numbered immediate actions with owner and timeline]
Under 250 words. Use CRITICAL / ALERT / WATCH / STABLE labels.

════════════════════════════════════════════════════════════════════════
  DETAILED ANALYST REPORT
════════════════════════════════════════════════════════════════════════

# ZaloPay Fraud Risk Report — {title}
**Period:** {start_date} → {end_date} | **Generated:** {generated_at}

## 1. KPI Summary
Headline metric + direction (MoM% or WoW%) + segment driver per relevant table.

## 2. Segment Analysis
One paragraph per segment with significant movement (>100M VND MoM / >20% WoW).

## 3. Investigation Priorities
Top 3: each with priority label, specific metric vs threshold, root cause hypothesis, next action with owner and timeline.

## 4. Promotion Abuse Status (if applicable)
%abuse metric + WoW direction + detection health.

## 5. Next Actions Table
Markdown table: | Priority | Action | Owner | Timeline |"""


def report_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()
    llm = get_llm(temperature=0.3, max_tokens=1000)

    findings      = state.get("findings") or []
    validation    = state.get("validation_result") or {}
    summaries     = state.get("summaries") or []
    analysis      = state.get("analysis_results") or {}
    dr            = state.get("date_range", {})
    report_type   = state.get("report_type", "weekly")
    pillar        = state.get("fraud_pillar", "general")
    generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        analysis_str = json.dumps(analysis, indent=2, default=str)[:1500]
    except Exception:
        analysis_str = str(analysis)[:1500]

    context = "\n\n".join([
        f"REPORT TYPE: {report_type} | PILLAR: {pillar}",
        f"PERIOD: {dr.get('start')} to {dr.get('end')}",
        f"VALIDATION: confidence={validation.get('confidence', 'N/A')}, validated={validation.get('validated')}",
        f"FINDINGS ({len(findings)}):\n" + json.dumps(findings, indent=2, default=str)[:1500],
        f"NARRATIVE SUMMARIES:\n" + "\n\n".join(summaries[:5]),
        f"ANALYSIS RESULTS (suggest_* outputs):\n{analysis_str}",
    ])

    title = f"{report_type.title()} | {pillar.replace('_', ' ').title()}"
    system = _REPORT_SYSTEM.format(
        title=title,
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
        generated_at=generated_at,
    )

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Generate the complete fraud report:\n\n{context}"),
    ])

    validation_banner = "✅ VALIDATED" if validation.get("validated") else "⚠️  PARTIAL VALIDATION"
    issues = validation.get("issues_found", [])
    issues_text = (
        "\n".join(f"  - [{i.get('severity','?').upper()}] {i.get('issue','')}" for i in issues)
        if issues else "  None"
    )

    final_report = f"""{resp.content.strip()}

{'═' * 72}
  VALIDATION STATUS : {validation_banner}
  CONFIDENCE        : {validation.get('confidence', 'N/A')}
  NOTES             : {validation.get('validation_notes', 'N/A')}
  ISSUES            :
{issues_text}
{'═' * 72}
"""

    logging.getLogger(__name__).info("TIMING report_node %.1fs", time.time() - _t0)
    return {
        "final_report": final_report,
        "messages": state.get("messages", [])
        + [{"role": "report", "content": "Final ZaloPay fraud report generated"}],
    }
