from __future__ import annotations
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm, structured_invoke, MAX_VALIDATION_RETRIES
from fraud_analytics.schemas.models import ValidationOutput

_SYSTEM = """You are the Validation Agent for a Fraud Analytics System.

Critically evaluate fraud findings for quality and accuracy.

Check each finding for:
  1. Specificity — are actual numbers cited? vague claims score low.
  2. Evidence sufficiency — is there data to back the claim?
  3. Contradictory signals — does any other data contradict the finding?
  4. Severity calibration — is the severity appropriate for the evidence?
  5. Actionability — is the recommended_action concrete and feasible?
  6. Historical consistency — does it align with known policies/precedent?

Decision rules:
  - If most findings are specific and well-supported → validated=true, next_step="report"
  - If findings lack data but queries haven't been exhaustive → validated=false, next_step="query"
  - If findings lack context/policy backing → validated=false, next_step="retrieval"
  - EXCEPTION: if retry_count >= {max_retries} → always set validated=true, next_step="report"

Current state:
  retry_count        : {retry_count} / {max_retries}
  findings_count     : {findings_count}
  queries_run        : {queries_run}
  documents_retrieved: {docs_count}"""


def validation_node(state: FraudReportState) -> Dict[str, Any]:
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_VALIDATION_RETRIES:
        validation_result = {
            "validated": True,
            "confidence": 0.75,
            "issues_found": [],
            "next_step": "report",
            "validation_notes": (
                f"Max retries ({MAX_VALIDATION_RETRIES}) reached. "
                "Proceeding to report generation with available findings."
            ),
        }
        return {
            "validation_result": validation_result,
            "retry_count": retry_count,
            "messages": state.get("messages", [])
            + [
                {
                    "role": "validation",
                    "content": "Max retries reached — forced pass → report",
                }
            ],
        }

    llm = get_llm(temperature=0.1)

    findings = state.get("findings") or []
    query_results = state.get("query_results") or {}
    docs = state.get("retrieved_documents") or []

    system_content = _SYSTEM.format(
        max_retries=MAX_VALIDATION_RETRIES,
        retry_count=retry_count,
        findings_count=len(findings),
        queries_run=list(query_results.keys()),
        docs_count=len(docs),
    )

    try:
        findings_json = json.dumps(findings, indent=2, default=str)[:4000]
    except Exception:
        findings_json = str(findings)[:4000]

    human_content = (
        f"Validate these {len(findings)} fraud findings:\n\n"
        f"{findings_json}\n\n"
        f"Available data sources: {list(query_results.keys())}\n"
        f"Retrieved documents: {len(docs)}"
    )

    result: ValidationOutput = structured_invoke(
        llm, [SystemMessage(content=system_content), HumanMessage(content=human_content)], ValidationOutput
    )

    if result is None:
        return {
            "validation_result": {
                "validated": True,
                "confidence": 0.6,
                "issues_found": [],
                "next_step": "report",
                "validation_notes": "Validation parsing failed — proceeding to report.",
            },
            "retry_count": MAX_VALIDATION_RETRIES,
            "messages": state.get("messages", [])
            + [{"role": "validation", "content": "PARSE FAILED — forced pass → report"}],
        }

    new_retry = retry_count + (0 if result.validated else 1)

    return {
        "validation_result": result.model_dump(),
        "retry_count": new_retry,
        "messages": state.get("messages", [])
        + [
            {
                "role": "validation",
                "content": (
                    f"{'PASSED' if result.validated else 'FAILED'} "
                    f"confidence={result.confidence:.2f} "
                    f"issues={len(result.issues_found)} "
                    f"next={result.next_step}"
                ),
            }
        ],
    }
