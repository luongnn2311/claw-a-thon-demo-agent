from __future__ import annotations
from typing import Dict, Any, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke


class ConversationDecision(BaseModel):
    action: Literal["clarify", "proceed", "follow_up", "answer", "end"] = Field(
        description="Next action"
    )
    message: str = Field(
        description=(
            "Message to show the user. "
            "Empty string when action is 'proceed' or 'answer'."
        )
    )


_SYSTEM = """You are the Conversation Manager for a ZaloPay Fraud Analytics chatbot.

Decide what to do next based on the conversation.

ACTIONS
  proceed     — Request is clear. Run the full fraud analysis pipeline now.
  clarify     — Request is vague. Ask exactly ONE short question.
  follow_up   — A report was just delivered and user has NOT yet been asked if they want details.
                Offer ONE short sentence inviting follow-up questions.
  answer      — User asked a specific question about the existing report
                (e.g. "why is X high?", "explain Y", "what does Z mean?",
                "show me appID breakdown", "more detail on international").
                Do NOT set a message — the followup node will answer.
  end         — User is done (says no/exit/quit/thanks/bye).

ROUTING RULES
  has_report = false:
    • Topic + time period identifiable                    → proceed
    • Either missing or ambiguous                         → clarify

  has_report = true, NO prior "follow_up" offer in history:
    • Always offer                                        → follow_up

  has_report = true, last assistant turn was a "follow_up" offer:
    • User asks a question or requests more detail        → answer
    • User requests a NEW/DIFFERENT analysis              → proceed
    • User says no / exit / done / thanks / bye           → end

  has_report = true, last assistant turn was an "answer":
    • User asks another question                          → answer
    • User requests a NEW/DIFFERENT analysis              → proceed
    • User says no / exit / done / thanks / bye           → end

MESSAGE CONTENT
  proceed     → "" (empty)
  answer      → "" (empty — followup node generates the answer)
  clarify     → brief question only
  follow_up   → one friendly sentence, e.g. "Would you like to dive deeper into any section?"
  end         → short farewell

Today: {today}
has_report: {has_report}

Conversation history (last 12 turns):
{history}

Latest user message: "{user_request}"
"""


def conversation_node(state: FraudReportState) -> Dict[str, Any]:
    llm = get_llm(temperature=0.2)

    today = datetime.now().strftime("%Y-%m-%d")
    history = list(state.get("conversation_history") or [])
    has_report = bool(state.get("final_report"))
    user_request = state.get("user_request", "")

    history_text = (
        "\n".join(f"  {m['role'].upper()}: {m['content']}" for m in history[-12:])
        or "  (none)"
    )

    result: ConversationDecision = structured_invoke(
        llm,
        [
            SystemMessage(content=_SYSTEM.format(
                today=today,
                has_report=has_report,
                history=history_text,
                user_request=user_request,
            )),
            HumanMessage(content="What should I do next?"),
        ],
        ConversationDecision,
    )

    updated_history = list(history)
    if result.message:
        updated_history.append({"role": "assistant", "content": result.message})

    updates: Dict[str, Any] = {
        "next_action": result.action,
        "agent_message": result.message,
        "conversation_history": updated_history,
    }

    # Wipe stale analysis only when starting a fresh pipeline run
    if result.action == "proceed":
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
