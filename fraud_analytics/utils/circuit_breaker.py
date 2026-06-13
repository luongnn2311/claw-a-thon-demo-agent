from __future__ import annotations
import time
import os
from typing import Dict, Any
from fraud_analytics.state import FraudReportState

MAX_NODE_VISITS: int = int(os.getenv("MAX_NODE_VISITS", "25"))
MAX_PIPELINE_SECONDS: int = int(os.getenv("MAX_PIPELINE_SECONDS", "120"))


def check_circuit_breaker(state: FraudReportState) -> bool:
    """Return True if any circuit breaker trips."""
    if state.get("total_node_visits", 0) >= MAX_NODE_VISITS:
        return True
    start = state.get("pipeline_start_time")
    if start and (time.time() - start) >= MAX_PIPELINE_SECONDS:
        return True
    return False


def increment_visits(state: FraudReportState) -> Dict[str, Any]:
    return {"total_node_visits": state.get("total_node_visits", 0) + 1}
