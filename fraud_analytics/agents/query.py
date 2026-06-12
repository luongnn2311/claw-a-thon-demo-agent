from __future__ import annotations
from typing import Dict, Any
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm
from fraud_analytics.tools.transaction_tools import (
    query_transaction_summary,
    query_discount_analysis,
    query_payment_solution_breakdown,
    query_trend_comparison,
    query_daily_volume_anomalies,
)
from fraud_analytics.tools.merchant_tools import (
    query_merchant_metrics,
    query_merchant_new_vs_existing,
)
from fraud_analytics.tools.user_tools import (
    query_user_metrics,
    query_user_discount_behavior,
)

ALL_TOOLS = [
    query_transaction_summary,
    query_discount_analysis,
    query_payment_solution_breakdown,
    query_trend_comparison,
    query_daily_volume_anomalies,
    query_merchant_metrics,
    query_merchant_new_vs_existing,
    query_user_metrics,
    query_user_discount_behavior,
]

_TOOL_MAP = {t.name: t for t in ALL_TOOLS}

_SYSTEM = """You are the Data Query Agent for a Fraud Analytics System.

Your job is to call the right data tools to gather evidence for the fraud analysis.

Available tools:
  - query_transaction_summary           : Overall volume, amounts, success/failure rates
  - query_discount_analysis             : Per-merchant discount patterns and abuse flags
  - query_payment_solution_breakdown    : Per-payment-solution metrics and failure rates
  - query_trend_comparison              : Current vs previous period comparison (requires 4 dates)
  - query_daily_volume_anomalies        : Z-score based daily volume anomaly detection
  - query_merchant_metrics              : Merchant outliers, concentration, top merchants
  - query_merchant_new_vs_existing      : New vs established merchant comparison
  - query_user_metrics                  : Active/new/repeat/suspicious users
  - query_user_discount_behavior        : Users with abnormal discount usage

Current context:
  report_type   : {report_type}
  fraud_pillar  : {fraud_pillar}
  date_range    : {start_date} to {end_date}
  previous period: {prev_start} to {prev_end}
  retry_count   : {retry_count}
  already_queried: {already_queried}

Rules:
- Always call query_transaction_summary for the current period.
- Call the tool(s) most relevant to fraud_pillar.
- Call query_trend_comparison to show period-over-period change.
- Do NOT call tools already in already_queried unless retry_count > 0.
- Stop calling tools once you have enough data (3-5 tool calls is usually sufficient)."""

_MAX_LOOP = 12


def _prev_period(start: str, end: str) -> tuple[str, str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s) + timedelta(days=1)
    pe = s - timedelta(days=1)
    ps = pe - delta + timedelta(days=1)
    return ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d")


def query_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    dr = state.get("date_range", {})
    start_date = dr.get("start", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
    end_date = dr.get("end", datetime.now().strftime("%Y-%m-%d"))
    prev_start, prev_end = _prev_period(start_date, end_date)

    existing_results: Dict[str, Any] = dict(state.get("query_results") or {})
    retry_count = state.get("retry_count", 0)

    system_content = _SYSTEM.format(
        report_type=state.get("report_type", "weekly"),
        fraud_pillar=state.get("fraud_pillar", "general"),
        start_date=start_date,
        end_date=end_date,
        prev_start=prev_start,
        prev_end=prev_end,
        retry_count=retry_count,
        already_queried=list(existing_results.keys()),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(
            content=(
                f"Gather data for {state.get('report_type', 'weekly')} "
                f"{state.get('fraud_pillar', 'general')} fraud analysis "
                f"from {start_date} to {end_date}."
            )
        ),
    ]

    new_results: Dict[str, Any] = {}

    for _ in range(_MAX_LOOP):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            break

        for tc in response.tool_calls:
            tool_fn = _TOOL_MAP.get(tc["name"])
            if tool_fn is None:
                result = f"Unknown tool: {tc['name']}"
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    result = f"Tool error: {exc}"
            new_results[tc["name"]] = result
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    merged = {**existing_results, **new_results}

    return {
        "query_results": merged,
        "messages": state.get("messages", [])
        + [
            {
                "role": "query",
                "content": f"Executed {len(new_results)} queries: {list(new_results.keys())}",
            }
        ],
    }