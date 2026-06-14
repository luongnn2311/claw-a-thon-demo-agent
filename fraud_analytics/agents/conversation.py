from __future__ import annotations
from typing import Dict, Any, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke


class ConversationDecision(BaseModel):
    action: Literal["clarify", "proceed", "answer", "end"] = Field(
        description="Next action"
    )
    message: str = Field(
        description=(
            "Message to show the user. "
            "Empty string when action is 'proceed' or 'answer'."
        )
    )


_PILLARS_HINT = """
Available report domains:
  • fraud_loss      — monthly + weekly loss by segment (domestic/international/VNPAY/…)
  • promo_abuse     — promo abuse rate, BAD_V2 / FAD detection effectiveness
  • coin2dd         — Coin-to-Direct-Debit abuse analysis
  • appid_breakdown — fraud breakdown by merchant / appID
  • general         — full overview across ALL domains above
"""

_SYSTEM = """You are the Conversation Manager for the ZaloPay Fraud Analytics Assistant.

You are a FRAUD ASSISTANT — not just a report generator.
Users can ask general fraud questions, explore domain knowledge, or request a full analysis report.

ACTIONS
  proceed  — User explicitly wants a full fraud analysis report generated.
             Only use this when the user clearly asks for a report/analysis AND the
             domain/pillar is known. The core function is report generation.
  answer   — Handle everything else: general questions, domain knowledge,
             follow-up questions about a previous report, data lookups,
             "what is X", "explain Y", "how does Z work", "show me the data".
             The followup agent will answer using knowledge + data tools + web search.
  clarify  — The request is ambiguous, OR the user wants a report but hasn't specified
             which domain/pillar. Recommend the available domains and ask ONE question.
  end      — User is done (says exit/quit/bye/thanks/no more).

ROUTING DECISION TREE

Step 1 — Is the user done?
  → exit / quit / bye / done / thanks / goodbye  →  end

Step 2 — Is it explicitly a REPORT REQUEST?
  Signals: "generate a report", "give me a report", "run analysis", "analyze X",
           "I need the weekly/monthly report", "show fraud report", "create report",
           "report on", "fraud analysis"
  → YES + pillar/domain is clear (fraud_loss / promo_abuse / coin2dd / appid_breakdown / general)
      AND date range is implied or stated  →  proceed
  → YES + pillar OR date range is unclear  →  clarify
    (Recommend the available domains listed below and ask which they want)

Step 3 — Everything else  →  answer
  This includes:
  - General questions: "what is BAD_V2?", "explain CNP fraud", "what is VAMP?"
  - Domain exploration: "how does Coin2DD abuse work?", "what are ZaloPay fraud segments?"
  - Data questions: "what was the fraud loss last week?", "show me appID breakdown"
  - Follow-up on existing report: "why is international high?", "tell me more about X"
  - Comparisons, thresholds, patterns, team ownership questions

IMPORTANT: Default to "answer" when in doubt. Only use "proceed" for clear, specific report requests.
{pillars_hint}
MESSAGE CONTENT
  proceed  → "" (empty)
  answer   → "" (empty — followup agent answers directly)
  clarify  → friendly question that recommends domains when relevant, one question only
  end      → short farewell

Today: {today}
has_report: {has_report}

Conversation history (last 12 turns):
{history}

Latest user message: "{user_request}"
"""

_WELCOME = """Hi! I'm the **ZaloPay Fraud Analytics Assistant** — your go-to tool for fraud data, patterns, and insights.

Here's what I can do for you:

**Generate a fraud report** for any of these domains:
- 💸 **Fraud Loss** — monthly/weekly loss by segment (domestic, international, VNPAY, …)
- 🎁 **Promo Abuse** — promo abuse rate, BAD_V2 / FAD detection effectiveness
- 🔄 **Coin2DD** — Coin-to-Direct-Debit abuse analysis
- 📱 **AppID Breakdown** — fraud broken down by merchant / appID
- 📊 **General** — full overview across all of the above

**Or just ask me anything** about ZaloPay fraud — patterns, thresholds, terminology, team ownership, or industry concepts.

What would you like to explore today?"""


def conversation_node(state: FraudReportState) -> Dict[str, Any]:
    today = datetime.now().strftime("%Y-%m-%d")
    history = list(state.get("conversation_history") or [])
    has_report = bool(state.get("final_report"))
    user_request = state.get("user_request", "").strip()

    # First entry — no history, no user message yet → show welcome
    if not history and not user_request:
        return {
            "next_action": "clarify",
            "agent_message": _WELCOME,
            "conversation_history": [{"role": "assistant", "content": _WELCOME}],
        }

    llm = get_llm(temperature=0.1)

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
                pillars_hint=_PILLARS_HINT,
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

    # Wipe stale pipeline state only when starting a fresh report run
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
