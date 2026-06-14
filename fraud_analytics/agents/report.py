from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

_EXEC_SYSTEM = """You are generating the Executive Summary for a ZaloPay Fraud Risk Report.

Audience: VP Risk, Head of Fraud Operations.

Format:
- Lead with the single most critical finding and its metric
- 4-6 bullet points with specific numbers (loss in M VND, %change, threshold comparison)
- End with exactly 3 recommended immediate actions (numbered, with owner and timeline)
- Under 300 words. Markdown format.

Use ZaloPay priority labels: CRITICAL / ALERT / WATCH / STABLE."""

_DETAIL_SYSTEM = """You are generating a ZaloPay Fraud Analytics Report for the fraud operations team.

# ZaloPay Fraud Risk Report — {title}
**Period:** {start_date} → {end_date} | **Generated:** {generated_at}

Produce these 5 sections in markdown using ZaloPay narrative templates:

## 1. KPI Summary
For each relevant table: headline metric + direction (MoM% or WoW%) + segment driver.
Use format: "Total fraud loss: X M VND (±Y% MoM), primarily driven by [segment]."

## 2. Segment Analysis
One paragraph per segment with significant movement (>100M VND MoM / >20% WoW).
Format: "[Segment] [dropped/increased] by ±X M VND MoM ([prev]M → [curr]M), driven by [appID/pattern]. ACR will [next action]."

## 3. Investigation Priorities
Top 3, each with:
  - Priority level (CRITICAL/ALERT/WATCH) and title
  - Specific metric vs threshold
  - Root cause hypothesis (reference known pattern)
  - Next action with owner and timeline

## 4. Promotion Abuse Status (if applicable)
%abuse metric + WoW direction + detection health note if degraded.

## 5. Next Actions Table
Markdown table: | Priority | Action | Owner | Timeline |"""


def report_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.4, max_tokens=2000)

    findings       = state.get("findings") or []
    validation     = state.get("validation_result") or {}
    summaries      = state.get("summaries") or []
    analysis       = state.get("analysis_results") or {}
    query_results  = state.get("query_results") or {}
    dr             = state.get("date_range", {})
    report_type    = state.get("report_type", "weekly")
    pillar         = state.get("fraud_pillar", "general")
    generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build context — prioritise analysis results + summaries over raw data
    try:
        analysis_str = json.dumps(analysis, indent=2, default=str)[:1500]
    except Exception:
        analysis_str = str(analysis)[:1500]

    context = "\n\n".join([
        f"REPORT TYPE: {report_type} | PILLAR: {pillar}",
        f"PERIOD: {dr.get('start')} to {dr.get('end')}",
        f"VALIDATION: confidence={validation.get('confidence', 'N/A')}, "
        f"validated={validation.get('validated')}",
        f"FINDINGS ({len(findings)}):\n"
        + json.dumps(findings, indent=2, default=str)[:1500],
        f"NARRATIVE SUMMARIES:\n" + "\n\n".join(summaries[:5]),
        f"ANALYSIS RESULTS (suggest_* outputs):\n{analysis_str}",
    ])

    title = f"{report_type.title()} | {pillar.replace('_', ' ').title()}"
    detail_system = _DETAIL_SYSTEM.format(
        title=title,
        start_date=dr.get("start", "N/A"),
        end_date=dr.get("end", "N/A"),
        generated_at=generated_at,
    )

    _req_count = 0

    def _exec():
        nonlocal _req_count
        t0 = time.time()
        _req_count += 1
        print(f"  [report] req#1 exec_summary START", flush=True)
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
        print(f"  [report] req#2 detail_report START", flush=True)
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

    validation_banner = "✅ VALIDATED" if validation.get("validated") else "⚠️  PARTIAL VALIDATION"
    issues = validation.get("issues_found", [])
    issues_text = (
        "\n".join(f"  - [{i.get('severity','?').upper()}] {i.get('issue','')}" for i in issues)
        if issues else "  None"
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
  CONFIDENCE        : {validation.get('confidence', 'N/A')}
  NOTES             : {validation.get('validation_notes', 'N/A')}
  ISSUES            :
{issues_text}
{'═' * 72}
"""

    return {
        "final_report": final_report,
        "messages": state.get("messages", [])
        + [{"role": "report", "content": "Final ZaloPay fraud report generated"}],
    }
