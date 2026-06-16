from __future__ import annotations
from typing import Dict, Any
from fraud_analytics.state import FraudReportState

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

_GREETINGS = {
    "hello", "hi", "hey", "xin chào", "chào", "chào bạn", "chào anh", "chào chị",
    "yo", "sup", "howdy", "good morning", "good afternoon", "good evening",
}

_QUIT_WORDS = {
    "exit", "quit", "bye", "goodbye", "stop", "end", "no", "done",
    "thoát", "kết thúc", "cảm ơn", "thanks", "thank you",
}


def conversation_node(state: FraudReportState) -> Dict[str, Any]:
    history = list(state.get("conversation_history") or [])
    user_request = state.get("user_request", "").strip()
    msg = user_request.lower().rstrip("!.,?")

    # First entry — no message yet → show welcome
    if not user_request and not history:
        return {
            "next_action": "clarify",
            "agent_message": _WELCOME,
            "conversation_history": [{"role": "assistant", "content": _WELCOME}],
        }

    # Greeting → show welcome
    if msg in _GREETINGS:
        updated = list(history)
        updated.append({"role": "assistant", "content": _WELCOME})
        return {
            "next_action": "clarify",
            "agent_message": _WELCOME,
            "conversation_history": updated,
        }

    # Exit
    if msg in _QUIT_WORDS:
        return {
            "next_action": "end",
            "agent_message": "Goodbye! Feel free to come back anytime.",
            "conversation_history": history,
        }

    # Everything else → route_node decides the path
    return {
        "next_action": "route",
        "agent_message": "",
        "conversation_history": history,
    }
