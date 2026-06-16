from __future__ import annotations
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke


class RouteDecision(BaseModel):
    route: Literal["full_pipeline", "query_data", "retrieve_knowledge"] = Field(
        description="Which execution path to take"
    )
    reason: str = Field(description="One sentence explaining the decision")


_SYSTEM = """You are a routing agent for a ZaloPay Fraud Analytics assistant.
Classify the user request into exactly one route.

full_pipeline
  User wants a complete fraud analysis report generated.
  Examples:
    "generate a fraud loss report"
    "give me the weekly promo report"
    "run monthly analysis for promo abuse"
    "I need a full report on coin2dd"
    "create a general fraud report"
    "monthly fraud loss analysis"

query_data
  User wants specific numbers, metrics, or data from the fraud database — no full report.
  Examples:
    "what was the fraud loss last week?"
    "show me the latest promo abuse rate"
    "how much fraud was there in the international segment?"
    "get me the appID breakdown data"
    "what are the current coin2dd figures?"
    "show me last month's promo numbers"

retrieve_knowledge
  User wants to understand a concept, pattern, threshold, term, or domain knowledge.
  Examples:
    "what is TPV?"
    "explain BAD_V2 detection"
    "how does Coin2DD abuse work?"
    "what are the fraud thresholds?"
    "tell me about promo abuse patterns"
    "why is international fraud high?"
    "define ATO"
    "what is VAMP?"
    "how does BAD_V2 differ from FAD?"

Rules:
- If the user says "query", "query out", "pull from", "retrieve from [file/table]" → always query_data
- If the user specifies their own output format (e.g., "tell me X, Y, Z") → query_data or retrieve_knowledge, NOT full_pipeline
- If the user asks for a REPORT or ANALYSIS → full_pipeline
- If the user asks for DATA, NUMBERS, FIGURES, METRICS, TRENDS → query_data
- If the user asks WHAT/WHY/HOW about a concept → retrieve_knowledge
- Default to retrieve_knowledge when uncertain

User request: {user_request}
"""


def route_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()

    user_request = state.get("user_request", "")
    llm = get_llm(temperature=0.0, max_tokens=150)

    result: RouteDecision = structured_invoke(
        llm,
        [
            SystemMessage(content=_SYSTEM.format(user_request=user_request)),
            HumanMessage(content="Classify this request."),
        ],
        RouteDecision,
    )

    if result is None:
        result = RouteDecision(route="retrieve_knowledge", reason="parse failure — defaulting to knowledge")

    logging.getLogger(__name__).info(
        "TIMING route_node %.1fs → %s | %s", time.time() - _t0, result.route, result.reason
    )

    updates: Dict[str, Any] = {"route_decision": result.route}

    # Reset pipeline state when entering the full report pipeline
    if result.route == "full_pipeline":
        updates.update({
            "final_report": "",
            "report_type": "",
            "fraud_pillar": "",
            "tables_to_use": [],
            "date_range": {},
            "retrieved_documents": [],
            "query_results": {},
            "analysis_results": {},
            "summaries": [],
            "findings": [],
            "validation_result": {},
            "retry_count": 0,
        })

    return updates
