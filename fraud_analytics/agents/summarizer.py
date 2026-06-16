"""
Summarizer node — converts structured suggest_* analysis results into
narrative paragraph summaries for each table, ready for the reasoning node.
"""
from __future__ import annotations
import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

_SYSTEM = """You are a ZaloPay Risk Analyst writing internal report summaries.

Convert the structured analysis findings below into concise narrative paragraphs
following ZaloPay's standard reporting style. For each table with findings:

1. State the headline metric (total_loss / %abuse / pct_abuse) with current value
2. State the direction (MoM or WoW change) and primary driver segment/campaign
3. List the top 1-2 priority actions using the exact priority labels (CRITICAL/ALERT/WATCH/STABLE)
4. Keep each paragraph under 100 words

Use specific numbers from the data. Do NOT use vague language.
Do NOT add findings that are not in the analysis results.

Report context: {report_type} | {fraud_pillar} | {date_range}

Analysis results:
{analysis_json}

Sample table data (last 2 rows per table):
{table_sample}"""


def summarizer_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()
    llm = get_llm(temperature=0.2, max_tokens=500)

    analysis_results = state.get("analysis_results") or {}
    query_results    = state.get("query_results") or {}
    dr = state.get("date_range", {})

    if not analysis_results:
        return {
            "summaries": ["No analysis results available to summarize."],
            "messages": state.get("messages", []) + [
                {"role": "summarizer", "content": "No analysis results — skipped."}
            ],
        }

    # Build sample table data (last 2 rows per table)
    table_sample: Dict[str, Any] = {}
    for tbl in ["fraud_monthly_loss", "fraud_weekly_loss", "promo_weekly_abuse",
                "coin2dd_monthly", "appid_fraud_breakdown"]:
        records = query_results.get(tbl, [])
        if records:
            table_sample[tbl] = records[-2:] if len(records) >= 2 else records

    try:
        analysis_json = json.dumps(analysis_results, indent=2, default=str)[:3000]
        sample_json   = json.dumps(table_sample, indent=2, default=str)[:2000]
    except Exception:
        analysis_json = str(analysis_results)[:3000]
        sample_json   = str(table_sample)[:2000]

    system_content = _SYSTEM.format(
        report_type=state.get("report_type", "weekly"),
        fraud_pillar=state.get("fraud_pillar", "general"),
        date_range=f"{dr.get('start', 'N/A')} to {dr.get('end', 'N/A')}",
        analysis_json=analysis_json,
        table_sample=sample_json,
    )

    response = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=(
            "Write one narrative paragraph per table that has findings. "
            "Include the priority level and specific numbers for each finding."
        )),
    ])

    raw = response.content.strip()
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    summaries: List[str] = paragraphs if paragraphs else [raw]

    logging.getLogger(__name__).info("TIMING summarizer_node %.1fs", time.time() - _t0)
    return {
        "summaries": summaries,
        "messages": state.get("messages", []) + [{
            "role": "summarizer",
            "content": f"Generated {len(summaries)} narrative summaries from {len(analysis_results)} analysis results.",
        }],
    }
