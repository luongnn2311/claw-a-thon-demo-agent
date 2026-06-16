from __future__ import annotations
from typing import TypedDict, List, Dict, Any


class FraudReportState(TypedDict, total=False):
    user_request: str
    report_type: str               # "weekly" | "monthly" | "adhoc"
    date_range: Dict[str, str]     # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    fraud_pillar: str              # "fraud_loss" | "promo_abuse" | "coin2dd" | "appid_breakdown" | "general"
    tables_to_use: List[str]       # subset of the 5 tables relevant to the request
    retrieved_documents: List[Dict[str, Any]]
    query_results: Dict[str, Any]  # raw pipeline output (5 tables as records)
    analysis_results: Dict[str, Any]  # suggest_* outputs per table
    summaries: List[str]           # narrative summaries produced by summarizer
    findings: List[Dict[str, Any]]
    validation_result: Dict[str, Any]
    final_report: str
    retry_count: int
    total_node_visits: int      # circuit breaker counter
    pipeline_start_time: float  # unix timestamp set by orchestrator
    messages: List[Dict[str, str]]
    # Conversation layer
    conversation_history: List[Dict[str, str]]  # full chat transcript
    next_action: str        # "route" | "clarify" | "end"
    route_decision: str     # "full_pipeline" | "query_data" | "retrieve_knowledge"
    agent_message: str      # question / farewell to display to the user
    validation_feedback: str  # targeted issues from validation_node for reasoning retry
