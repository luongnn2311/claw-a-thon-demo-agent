from __future__ import annotations
from typing import TypedDict, List, Dict, Any


class FraudReportState(TypedDict, total=False):
    user_request: str
    report_type: str               # "weekly" | "monthly" | "adhoc"
    date_range: Dict[str, str]     # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    fraud_pillar: str              # e.g. "merchant_abuse", "discount_abuse"
    retrieved_documents: List[Dict[str, Any]]
    query_results: Dict[str, Any]
    summaries: List[str]
    findings: List[Dict[str, Any]]
    validation_result: Dict[str, Any]
    final_report: str
    retry_count: int
    total_node_visits: int      # circuit breaker counter
    pipeline_start_time: float  # unix timestamp set by orchestrator
    messages: List[Dict[str, str]]
    # Conversation layer
    conversation_history: List[Dict[str, str]]  # full chat transcript
    next_action: str        # "proceed" | "clarify" | "follow_up" | "end"
    agent_message: str      # question / farewell to display to the user
