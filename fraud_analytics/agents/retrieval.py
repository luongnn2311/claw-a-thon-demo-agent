from __future__ import annotations
from typing import Dict, Any, Optional
from fraud_analytics.state import FraudReportState
from fraud_analytics.knowledge.vector_store import FraudKnowledgeBase
from fraud_analytics.config import MAX_RETRIEVAL_DOCS, VECTOR_STORE_PATH

_kb: Optional[FraudKnowledgeBase] = None


def _get_kb() -> FraudKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = FraudKnowledgeBase(persist_path=VECTOR_STORE_PATH)
    return _kb


def retrieval_node(state: FraudReportState) -> Dict[str, Any]:
    kb = _get_kb()

    pillar = state.get("fraud_pillar", "general")
    report_type = state.get("report_type", "weekly")
    user_request = state.get("user_request", "")

    queries = [
        f"{pillar} thresholds alert triggers investigation",
        f"{pillar} narrative templates decision tree",
        f"{pillar} fraud patterns known ZaloPay",
        user_request[:200],
    ]

    seen: set = set()
    docs: list = []
    for q in queries:
        for doc in kb.search(q, k=MAX_RETRIEVAL_DOCS):
            key = doc["content"][:80]
            if key not in seen:
                seen.add(key)
                docs.append(doc)

    docs_to_keep = docs[: MAX_RETRIEVAL_DOCS * 2]

    return {
        "retrieved_documents": docs_to_keep,
        "messages": state.get("messages", [])
        + [
            {
                "role": "retrieval",
                "content": f"Retrieved {len(docs_to_keep)} knowledge documents for pillar={pillar}",
            }
        ],
    }
