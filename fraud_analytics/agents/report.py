from __future__ import annotations
import json
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from fraud_analytics.state import FraudReportState
from fraud_analytics.config import get_llm

# ── Template: Weekly Fraud Loss ───────────────────────────────────────────────
_WEEKLY_FRAUD_TEMPLATE = """You are generating a ZaloPay Weekly Fraud Loss Report.
Use EXACTLY this structure — no extra sections, no deviation:

════════════════════════════════════════════════════════════════════════
  WEEKLY FRAUD LOSS REPORT — {title}
════════════════════════════════════════════════════════════════════════

**HEADLINE**
Week of [date]: total fraud loss [X]M VND, [up/down] [%] WoW ([prev]M → [curr]M).
Primary driver: [segment] [direction] driven by [appID/pattern].
[Alert level: CRITICAL / ALERT / WATCH / STABLE] — [action required if any].

**SEGMENT BREAKDOWN**
For each segment with notable movement, one line each:
  [Segment]: [X]M VND ([±%] WoW) — [brief reason/pattern]

**INVESTIGATION PRIORITIES**

PRIORITY 1 — [CRITICAL/ALERT/WATCH]: [title]
  Data: [specific metric] = [value] vs threshold [threshold value]
  Root cause hypothesis: [pattern]
  Next action: [specific operational action with owner and timeline]

PRIORITY 2 — [level]: [title]
  Data: ...
  Root cause hypothesis: ...
  Next action: ...

PRIORITY 3 — [level]: [title]
  Data: ...
  Root cause hypothesis: ...
  Next action: ...
"""

# ── Template: Monthly Fraud Loss ──────────────────────────────────────────────
_MONTHLY_FRAUD_TEMPLATE = """You are generating a ZaloPay Monthly Fraud Loss Report.
Use EXACTLY this structure:

════════════════════════════════════════════════════════════════════════
  MONTHLY FRAUD LOSS REPORT — {title}
════════════════════════════════════════════════════════════════════════

**HEADLINE**
TotalLoss recorded at [X]B VND ([±%] MoM), [decreased/increased] significantly by [±delta]B VND MoM
([prev]B → [curr]B), primarily driven by [improvements/deterioration] in [main segment].

**SEGMENT ANALYSIS**
For each segment with significant movement (>100M VND MoM):

[Segment — IMPROVING]:
[Segment] dropped by [±delta]M VND MoM ([prev]M → [curr]M).
Controls deployed on [control target] continued to perform effectively.
However, [remaining risk or residual pattern].
In next month, ACR will [next action].

[Segment — WORSENING]:
[Segment] increased by [+delta]M VND MoM ([prev]M → [curr]M),
primarily driven by [appID/BIN/pattern].
ACR will [specific next action] to address [specific risk].

**INVESTIGATION PRIORITIES**

PRIORITY 1 — [CRITICAL/ALERT/WATCH]: [title]
  Data: [specific metric] = [value] vs threshold [threshold value]
  Root cause hypothesis: [pattern]
  Next action: [specific operational action with owner and timeline]

PRIORITY 2 — [level]: [title]
  Data: ...
  Root cause hypothesis: ...
  Next action: ...

PRIORITY 3 — [level]: [title]
  Data: ...
  Root cause hypothesis: ...
  Next action: ...
"""

# ── Template: Weekly Promo Abuse ──────────────────────────────────────────────
_WEEKLY_PROMO_TEMPLATE = """You are generating a ZaloPay Weekly Promo Abuse Report.
Use EXACTLY this structure:

════════════════════════════════════════════════════════════════════════
  WEEKLY PROMO ABUSE REPORT — {title}
════════════════════════════════════════════════════════════════════════

**HEADLINE**
Week of [date]: totalSpending=[X]M VND, totalAbuse=[Y]M VND, %abuse=[Z]%.
[Increasing/decreasing/stable] compared to prior week ([WoW%] change).
[Main driver if spike]: primarily driven by [campaign/appID/pattern].
[If pct_abuse < 1.5%]: Note: unusually low — likely visibility loss, not improvement.

**DETECTION HEALTH**
BAD_V2: [status — active/degraded/dropped]
FAD: [status]
[If both declining]: Detection degraded since [date]. Manual review required.

**INVESTIGATION PRIORITIES**

PRIORITY 1 — [RED FLAG/ALERT/WATCH]: [campaign or pattern name]
  %abuse: [value]% vs threshold [3%/4%]
  Top campaign: [campaignCode] — [amount] abuse
  Detection source status: [BAD_V2 / FAD health]
  Next action: [specific action with owner — RPO PRE / DS team]

PRIORITY 2 — [level]: [title]
  ...

PRIORITY 3 — [level]: [title]
  ...
"""

# ── Template: General / Coin2DD / AppID / Fallback ────────────────────────────
_GENERAL_TEMPLATE = """You are generating a ZaloPay Fraud Analytics Report.
Use EXACTLY this structure:

════════════════════════════════════════════════════════════════════════
  FRAUD ANALYTICS REPORT — {title}
════════════════════════════════════════════════════════════════════════

**EXECUTIVE SUMMARY**
[Lead with the single most critical finding + metric. 3–5 bullets with specific numbers.]
[Exactly 3 numbered immediate actions with owner and timeline.]

**KEY FINDINGS**
For each domain with CRITICAL or ALERT findings:

[Domain — CRITICAL/ALERT]:
[Specific metric] = [value] vs threshold [value]. [Brief narrative.]
Next action: [specific operational action with owner and timeline]

**INVESTIGATION PRIORITIES**

PRIORITY 1 — [CRITICAL/ALERT/WATCH]: [title]
  Data: [specific metric] = [value] vs threshold [threshold value]
  Root cause hypothesis: [pattern]
  Next action: [specific operational action with owner and timeline]

PRIORITY 2 — ...
PRIORITY 3 — ...
"""


def _pick_template(report_type: str, pillar: str) -> str:
    if pillar == "promo_abuse":
        return _WEEKLY_PROMO_TEMPLATE
    if pillar in ("fraud_loss", "appid_breakdown"):
        if report_type == "monthly":
            return _MONTHLY_FRAUD_TEMPLATE
        return _WEEKLY_FRAUD_TEMPLATE
    return _GENERAL_TEMPLATE


def report_node(state: FraudReportState) -> Dict[str, Any]:
    import time, logging
    _t0 = time.time()
    llm = get_llm(temperature=0.3, max_tokens=1000)

    findings      = state.get("findings") or []
    validation    = state.get("validation_result") or {}
    summaries     = state.get("summaries") or []
    analysis      = state.get("analysis_results") or {}
    dr            = state.get("date_range", {})
    report_type   = state.get("report_type", "weekly")
    pillar        = state.get("fraud_pillar", "general")
    generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        analysis_str = json.dumps(analysis, indent=2, default=str)[:1500]
    except Exception:
        analysis_str = str(analysis)[:1500]

    context = "\n\n".join([
        f"REPORT TYPE: {report_type} | PILLAR: {pillar}",
        f"PERIOD: {dr.get('start')} to {dr.get('end')}",
        f"VALIDATION: confidence={validation.get('confidence', 'N/A')}, validated={validation.get('validated')}",
        f"FINDINGS ({len(findings)}):\n" + json.dumps(findings, indent=2, default=str)[:1500],
        f"NARRATIVE SUMMARIES:\n" + "\n\n".join(summaries[:5]),
        f"ANALYSIS RESULTS (suggest_* outputs):\n{analysis_str}",
    ])

    title = f"{report_type.title()} | {pillar.replace('_', ' ').title()} | {dr.get('start', 'N/A')} → {dr.get('end', 'N/A')}"
    system = _pick_template(report_type, pillar).format(title=title)

    resp = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Generate the report using the data below:\n\n{context}"),
    ])

    validation_banner = "✅ VALIDATED" if validation.get("validated") else "⚠️  PARTIAL VALIDATION"
    issues = validation.get("issues_found", [])
    issues_text = (
        "\n".join(f"  - [{i.get('severity','?').upper()}] {i.get('issue','')}" for i in issues)
        if issues else "  None"
    )

    final_report = f"""{resp.content.strip()}

{'═' * 72}
  VALIDATION : {validation_banner}
  CONFIDENCE : {validation.get('confidence', 'N/A')}
  NOTES      : {validation.get('validation_notes', 'N/A')}
  ISSUES     :
{issues_text}
{'═' * 72}
  Generated  : {generated_at}
{'═' * 72}
"""

    logging.getLogger(__name__).info("TIMING report_node %.1fs", time.time() - _t0)
    return {
        "final_report": final_report,
        "messages": state.get("messages", [])
        + [{"role": "report", "content": "Final ZaloPay fraud report generated"}],
    }
