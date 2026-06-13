from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
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

_DETAIL_SYSTEM = """You are generating a Fraud Analytics Report for the fraud operations team.

# Fraud Analytics Report — {title}
**Period:** {start_date} → {end_date} | **Generated:** {generated_at}

Produce these 5 sections in markdown. Be concise — 2-4 sentences per section max.

## 1. Overview
Key volume, amount, success rate metrics only.

## 2. Key Findings
Numbered list with severity badge (🔴🟠🟡🟢) and one sentence each.

## 3. Fraud Indicators
Bullet points per pillar with the single most important metric.

## 4. Risk Assessment
Markdown table: Pillar | Likelihood | Impact | Overall

## 5. Top 3 Recommendations
Numbered. One action sentence each with owner and timeline."""


def report_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.4, max_tokens=2000)

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
            f"FINDINGS ({len(findings)}):\n{json.dumps(findings, indent=2, default=str)[:1500]}",
            f"DATA SUMMARIES:\n" + "\n".join(summaries[:5]),
        ]
    )

    title = f"{report_type.title()} | {pillar.replace('_', ' ').title()}"
    detail_system = _DETAIL_SYSTEM.format(
        title=title,
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
        generated_at=generated_at,
    )

    import time
    _req_count = 0

    def _exec():
        nonlocal _req_count
        t0 = time.time()
        _req_count += 1
        print(f"  [report] req#{_req_count} exec_summary START", flush=True)
        r = llm.invoke([
            SystemMessage(content=_EXEC_SYSTEM),
            HumanMessage(content=f"Generate executive summary:\n\n{context}"),
        ])
        print(f"  [report] req#1 exec_summary DONE in {time.time()-t0:.1f}s", flush=True)
        return r

    def _detail():
        nonlocal _req_count
        t0 = time.time()
        _req_count += 1
        print(f"  [report] req#{_req_count} detail_report START", flush=True)
        r = llm.invoke([
            SystemMessage(content=detail_system),
            HumanMessage(content=f"Generate the complete analyst report:\n\n{context}"),
        ])
        print(f"  [report] req#2 detail_report DONE in {time.time()-t0:.1f}s", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_exec = pool.submit(_exec)
        f_detail = pool.submit(_detail)
        exec_resp = f_exec.result()
        detail_resp = f_detail.result()

    print(f"  [report] total LLM requests: {_req_count}", flush=True)

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
