"""
Multi-Agent Fraud Analytics & Reporting System
===============================================
LangGraph-based pipeline with 7 specialized agents + conversational loop.

Usage:
    python main.py                          # multi-turn chat (default)
    python main.py --single "your request" # one-shot, no conversation
    python main.py --draw-graph             # save graph diagram to output/
"""
from __future__ import annotations
import os
import sys
from datetime import datetime


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_report(report: str) -> str:
    os.makedirs("output", exist_ok=True)
    filename = f"output/fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    return filename


def draw_graph() -> None:
    from fraud_analytics.graph.fraud_graph import build_fraud_graph
    graph = build_fraud_graph()
    try:
        png = graph.get_graph().draw_mermaid_png()
        path = "output/fraud_graph.png"
        os.makedirs("output", exist_ok=True)
        with open(path, "wb") as f:
            f.write(png)
        print(f"Graph diagram saved to {path}")
    except Exception as e:
        print(f"Could not render PNG ({e}). Mermaid source:\n")
        print(graph.get_graph().draw_mermaid())


# ── Single-run mode ───────────────────────────────────────────────────────────

def run_fraud_analysis(user_request: str, verbose: bool = True) -> str:
    """One-shot: single request → report → done. No conversation loop."""
    from fraud_analytics.graph.fraud_graph import build_fraud_graph

    if verbose:
        print(f"\n{'═' * 60}")
        print(f"REQUEST : {user_request}")
        print(f"{'═' * 60}\n")
        print("Building multi-agent graph ...", flush=True)

    graph = build_fraud_graph()
    initial_state = {
        "user_request": user_request,
        "retry_count": 0,
        "messages": [],
    }

    if verbose:
        print("Running pipeline ...\n")

    final_state = graph.invoke(initial_state, config={"recursion_limit": 30})

    if verbose:
        print("\nAgent execution trace:")
        for msg in final_state.get("messages", []):
            role = msg.get("role", "?").upper().ljust(12)
            print(f"  [{role}] {msg.get('content', '')}")

    return final_state.get("final_report", "No report generated.")


# ── Multi-turn chat mode ──────────────────────────────────────────────────────

def run_chat() -> None:
    """
    Multi-turn conversational mode.

    The main agent decides after every user message:
      proceed    → run full fraud analysis pipeline
      clarify    → ask user a focused question before running
      follow_up  → after a report, offer to do more
      end        → gracefully close the session
    """
    from fraud_analytics.graph.fraud_graph import build_chat_graph
    from langgraph.types import Command

    graph, config = build_chat_graph()

    print("\nFraud Analytics Agent")
    print("─" * 42)
    print("Describe what you want to analyze.")
    print("Type 'exit' or 'quit' to end the session.\n")

    first_input = input("You: ").strip()
    if not first_input or first_input.lower() in ("exit", "quit", "bye"):
        return

    initial_state = {
        "user_request": first_input,
        "conversation_history": [{"role": "user", "content": first_input}],
        "retry_count": 0,
        "messages": [],
    }

    seen_report = ""

    # ── First invocation ──────────────────────────────────────────────────────
    state = graph.invoke(initial_state, config)

    while True:
        # Show any new report
        report = state.get("final_report") or ""
        if report and report != seen_report:
            print(report)
            path = save_report(report)
            print(f"Report saved → {path}\n")
            seen_report = report

        # Check if graph finished (hit END via "end" action)
        snapshot = graph.get_state(config)
        if not snapshot.next:
            farewell = state.get("agent_message", "")
            if farewell:
                print(f"\nAgent: {farewell}")
            break

        # Graph is paused at human_input — show agent's message and wait
        agent_msg = snapshot.values.get("agent_message", "")
        if agent_msg:
            print(f"\nAgent: {agent_msg}")

        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit", "bye"):
            print("\nAgent: Session ended. Goodbye!")
            break

        # Resume the graph with the user's response
        state = graph.invoke(Command(resume=user_input), config)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if "--draw-graph" in args:
        draw_graph()
        return

    # Single-shot mode: python main.py --single "your request"
    if "--single" in args:
        idx = args.index("--single")
        remaining = args[idx + 1:]
        if not remaining:
            print("Usage: python main.py --single \"your request\"")
            sys.exit(1)
        user_request = " ".join(remaining)
        report = run_fraud_analysis(user_request)
        print("\n" + "═" * 60)
        print(report)
        print("═" * 60)
        path = save_report(report)
        print(f"\nReport saved → {path}")
        return

    # Default: multi-turn chat
    run_chat()


if __name__ == "__main__":
    main()
