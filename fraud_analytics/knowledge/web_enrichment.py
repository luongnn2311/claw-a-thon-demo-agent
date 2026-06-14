"""
Web enrichment utility — searches the internet for fintech/fraud risk context
and returns clean snippets as supplementary knowledge documents.

RULES:
- Call ONLY when existing knowledge is insufficient to answer the question
- Results are SUPPLEMENTARY context — never quote them verbatim to the user
- Search scope is always anchored to the fintech / payment fraud domain
- Always labeled source=web_search so reasoning nodes know their origin
"""
from __future__ import annotations
import re
from typing import List, Dict, Any

_DOMAIN_ANCHOR = "fintech payment fraud risk"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:500]


def search_web(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search DuckDuckGo for fintech/fraud-domain content related to `query`.
    Returns a list of document dicts (same format as retrieved_documents).
    Returns [] silently on any error.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return []

    anchored = f"{query} {_DOMAIN_ANCHOR}"
    docs: List[Dict[str, Any]] = []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(anchored, max_results=max_results, safesearch="off"))
    except Exception:
        return []

    for r in results:
        body = _clean(r.get("body", ""))
        title = (r.get("title") or "")[:120]
        url = r.get("href", "")
        if not body:
            continue
        docs.append({
            "content": f"{title}\n{body}",
            "metadata": {
                "source": "web_search",
                "type": "external",
                "url": url,
                "query": query,
            },
            "relevance_score": 0.5,
        })

    return docs
