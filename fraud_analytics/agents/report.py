from __future__ import annotations
import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

_EXEC_SYSTEM = """You are generating the Executive Summary for a Fraud Risk Report.

Audience: Senior management, VP Risk, Head of Fraud Operations.

Requirements:
- Under 300 words, punchy and direct
- Lead with the single most critical risk
- Include 3-5 key metrics as bullet points
- End with exactly 3 recommended immediate actions numbered 1-3
- Use plain business language, no jargon
- Format as markdown"""

_DETAIL_SYSTEM = """You are generating a complete Fraud Analytics Report for the fraud operations team.

Audience: Fraud analysts, risk operations, compliance.

You MUST produce all 7 sections in markdown. Use specific numbers everywhere.

# Fraud Analytics Report — {title}
**Period:** {start_date} → {end_date} | **Generated:** {generated_at}

---

## 1. Overview
(Transaction volume, total amount, success rate, key period metrics)

## 2. Key Findings
(Numbered list with severity badges: 🔴 Critical | 🟠 High | 🟡 Medium | 🟢 Low)

## 3. Fraud Indicators by Pillar
(Organized by fraud dimension with supporting metrics)

## 4. Supporting Evidence
(Specific data points, percentages, Z-scores, thresholds breached)

## 5. Risk Assessment
(Risk matrix table — pillar × likelihood × impact × overall)

## 6. Recommendations
(Minimum 5 specific, actionable operational steps with owner and timeline)

## 7. Appendix — Key Query Results
(2-3 important raw figures from the data queries)"""


def report_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.4)

    findings = state.get("findings") or []
    validation = state.get("validation_result") or {}
    query_results = state.get("query_results") or {}
    summaries = state.get("summaries") or []
    dr = state.get("date_range", {})
    report_type = state.get("report_type", "weekly")
    pillar = state.get("fraud_pillar", "general")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context = "\n\n".join(
        [
            f"REPORT TYPE: {report_type} | PILLAR: {pillar}",
            f"PERIOD: {dr.get('start')} to {dr.get('end')}",
            f"VALIDATION CONFIDENCE: {validation.get('confidence', 'N/A')}",
            f"FINDINGS ({len(findings)}):\n{json.dumps(findings, indent=2, default=str)[:3000]}",
            f"DATA SUMMARIES:\n" + "\n".join(summaries[:8]),
            f"QUERY SNAPSHOT:\n{json.dumps(dict(list(query_results.items())[:3]), indent=2, default=str)[:2000]}",
        ]
    )

    exec_resp = llm.invoke(
        [
            SystemMessage(content=_EXEC_SYSTEM),
            HumanMessage(content=f"Generate executive summary:\n\n{context}"),
        ]
    )

    title = f"{report_type.title()} | {pillar.replace('_', ' ').title()}"
    detail_system = _DETAIL_SYSTEM.format(
        title=title,
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
        generated_at=generated_at,
    )
    detail_resp = llm.invoke(
        [
            SystemMessage(content=detail_system),
            HumanMessage(content=f"Generate the complete analyst report:\n\n{context}"),
        ]
    )

    validation_banner = (
        "✅ VALIDATED" if validation.get("validated") else "⚠️  PARTIAL VALIDATION"
    )
    confidence = validation.get("confidence", "N/A")
    issues = validation.get("issues_found", [])
    issues_text = (
        "\n".join(f"  - [{i.get('severity','?').upper()}] {i.get('issue','')}" for i in issues)
        if issues
        else "  None"
    )

    final_report = f"""
{'═' * 72}
  EXECUTIVE SUMMARY
{'═' * 72}

{exec_resp.content.strip()}

{'═' * 72}
  DETAILED ANALYST REPORT
{'═' * 72}

{detail_resp.content.strip()}

{'═' * 72}
  VALIDATION STATUS : {validation_banner}
  CONFIDENCE        : {confidence}
  VALIDATION NOTES  : {validation.get('validation_notes', 'N/A')}
  ISSUES FLAGGED    :
{issues_text}
{'═' * 72}
"""

    return {
        "final_report": final_report,
        "messages": state.get("messages", [])
        + [{"role": "report", "content": "Final report generated"}],
    }
