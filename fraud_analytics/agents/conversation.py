from __future__ import annotations
from typing import Dict, Any, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke


class ConversationDecision(BaseModel):
    action: Literal["clarify", "proceed", "follow_up", "end"] = Field(
        description="Next action"
    )
    message: str = Field(
        description=(
            "Message to show the user. "
            "Empty string when action is 'proceed'."
        )
    )


_SYSTEM = """You are the Conversation Manager for a Fraud Analytics chatbot.

Decide what to do next based on the conversation.

ACTIONS
  proceed    — Request is clear. Run the fraud analysis pipeline now.
  clarify    — Request is vague. Ask exactly ONE short question to gather missing info.
  follow_up  — A report was just delivered. Offer in one sentence to do more.
  end        — User is done. Reply with a short goodbye.

RULES
  has_report = false:
    • Topic + time period are identifiable              → proceed
    • Either is missing or ambiguous                    → clarify

  has_report = true, no prior follow-up in history:
    • Always                                            → follow_up

  has_report = true, last assistant message was a follow-up:
    • User asks for another analysis                    → proceed
    • User says no / done / thanks / bye                → end

MESSAGE
  proceed    → "" (empty)
  clarify    → brief question
  follow_up  → one friendly sentence
  end        → short farewell

Today: {today}
has_report: {has_report}

Conversation history:
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
            SystemMessage(
                content=_SYSTEM.format(
                    today=today,
                    has_report=has_report,
                    history=history_text,
                    user_request=user_request,
                )
            ),
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

    # Wipe stale analysis so the pipeline runs fresh for a new request
    if result.action == "proceed":
        updates.update(
            {
                "final_report": "",
                "report_type": "",
                "fraud_pillar": "",
                "date_range": {},
                "retrieved_documents": [],
                "query_results": {},
                "summaries": [],
                "findings": [],
                "validation_result": {},
                "retry_count": 0,
            }
        )

    return updates
