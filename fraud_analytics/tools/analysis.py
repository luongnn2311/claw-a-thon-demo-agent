"""
Deterministic suggest_* functions translated from domain knowledge.
Each function returns a list of {"priority": str, "message": str} dicts.
Priority levels: CRITICAL, ALERT, WATCH, STABLE, CONFIRM, INVESTIGATE, NOTE
"""
from __future__ import annotations
from typing import List, Dict, Any


def _row(records: List[Dict], idx: int) -> Dict:
    if idx < len(records):
        return records[idx]
    return {}


def analyze_fraud_monthly(records: List[Dict]) -> List[Dict[str, str]]:
    """Analyze fraud_monthly_loss table. Compares latest month vs prior month."""
    if len(records) < 2:
        return [{"priority": "NOTE", "message": "Need at least 2 months of data for MoM analysis."}]

    curr = records[-1]
    prev = records[-2]
    suggestions = []

    mom_pct = curr.get("MoM_pct_change", 0) or 0

    if mom_pct > 40:
        suggestions.append({"priority": "CRITICAL",
            "message": f"Total fraud loss increased +{mom_pct:.0f}% MoM to {curr.get('total_loss', 0):.0f}M VND. Immediate investigation required."})
    elif mom_pct > 20:
        suggestions.append({"priority": "ALERT",
            "message": f"Total fraud loss increased +{mom_pct:.0f}% MoM. Identify segment driver and deploy rule within 48h."})
    elif mom_pct < -20:
        suggestions.append({"priority": "CONFIRM",
            "message": f"Total fraud loss declined {mom_pct:.0f}% MoM to {curr.get('total_loss', 0):.0f}M VND. Confirm controls are holding; check for detection gaps."})

    # International segment
    intl_change = (curr.get("international_loss", 0) or 0) - (prev.get("international_loss", 0) or 0)
    if intl_change > 200:
        suggestions.append({"priority": "INVESTIGATE",
            "message": f"International loss +{intl_change:.0f}M VND MoM. Check appIDs: 149 (Mobile Payment), 3762 (VNGGameShop), 9999 (Apple). Deploy targeted appID rule or BIN-level limit."})
    elif intl_change < -200:
        suggestions.append({"priority": "CONFIRM",
            "message": f"International loss {intl_change:.0f}M VND MoM. Verify controls on gaming/telco appIDs are holding."})

    # Domestic Direct segment
    dd_change = (curr.get("domestic_direct_loss", 0) or 0) - (prev.get("domestic_direct_loss", 0) or 0)
    if dd_change > 100:
        suggestions.append({"priority": "INVESTIGATE",
            "message": f"Domestic Direct +{dd_change:.0f}M VND MoM. Check appID 454 VCB sub-segment. Suggest granular behavioral controls or tighter per-card velocity limits."})

    # Domestic Napas segment
    nap_change = (curr.get("domestic_napas_loss", 0) or 0) - (prev.get("domestic_napas_loss", 0) or 0)
    if nap_change > 100:
        suggestions.append({"priority": "INVESTIGATE",
            "message": f"Domestic Napas +{nap_change:.0f}M VND MoM. Check BIN concentration and SME merchant overlap. Block high-risk BIN + deactivate fraud-flagged merchants."})

    # Concentration check
    total = curr.get("total_loss", 0) or 0
    if total > 0:
        segments = {
            "domestic_direct": curr.get("domestic_direct_loss", 0) or 0,
            "international": curr.get("international_loss", 0) or 0,
            "domestic_napas": curr.get("domestic_napas_loss", 0) or 0,
        }
        top_seg = max(segments, key=lambda k: segments[k])
        top_pct = segments[top_seg] / total * 100
        if top_pct > 60:
            suggestions.append({"priority": "NOTE",
                "message": f"{top_seg} is highly concentrated at {top_pct:.0f}% of total loss. Further granular segmentation recommended."})

    if not suggestions:
        suggestions.append({"priority": "STABLE",
            "message": f"Fraud monthly loss at {curr.get('total_loss', 0):.0f}M VND, {mom_pct:+.1f}% MoM — within normal range."})

    return suggestions


def analyze_fraud_weekly(records: List[Dict]) -> List[Dict[str, str]]:
    """Analyze fraud_weekly_loss table. Compares latest week vs prior week."""
    if len(records) < 2:
        return [{"priority": "NOTE", "message": "Need at least 2 weeks of data for WoW analysis."}]

    curr = records[-1]
    prev = records[-2]
    suggestions = []

    wow_pct = curr.get("WoW_pct_change", 0) or 0

    if wow_pct > 50:
        suggestions.append({"priority": "CRITICAL",
            "message": f"Weekly fraud jumped +{wow_pct:.0f}% WoW to {curr.get('total_loss', 0):.0f}M VND. Investigate same-day with ACR."})
    elif wow_pct > 20:
        suggestions.append({"priority": "ALERT",
            "message": f"Weekly fraud +{wow_pct:.0f}% WoW. Identify segment driver and deploy rule within 48h."})
    elif wow_pct < -30:
        suggestions.append({"priority": "CONFIRM",
            "message": f"Weekly fraud dropped {wow_pct:.0f}% WoW. Verify this is real improvement and not a detection gap."})

    # International spike
    intl_curr = curr.get("international_loss", 0) or 0
    intl_prev = prev.get("international_loss", 0) or 0
    if intl_prev > 0 and intl_curr > intl_prev * 1.3:
        suggestions.append({"priority": "INVESTIGATE",
            "message": "International segment driving WoW increase. Check appIDs 149, 3762 for new attack patterns."})

    # Domestic Direct spike
    dd_curr = curr.get("domestic_direct_loss", 0) or 0
    dd_prev = prev.get("domestic_direct_loss", 0) or 0
    if dd_prev > 0 and dd_curr > dd_prev * 1.3:
        suggestions.append({"priority": "INVESTIGATE",
            "message": "Domestic Direct segment rising. Check appID 454 VCB sub-segment and device/card signals."})

    if not suggestions:
        suggestions.append({"priority": "STABLE",
            "message": f"Weekly fraud at {curr.get('total_loss', 0):.2f}M VND, {wow_pct:+.1f}% WoW — within normal range."})

    return suggestions


def analyze_promo_weekly(records: List[Dict], detection_healthy: bool = True) -> List[Dict[str, str]]:
    """Analyze promo_weekly_abuse table. Evaluates latest week."""
    if not records:
        return [{"priority": "NOTE", "message": "No promo weekly data available."}]

    curr = records[-1]
    prev = records[-2] if len(records) >= 2 else {}
    suggestions = []

    pct = curr.get("pct_abuse", 0) or 0
    wow_pct = curr.get("abuse_trend", 0) or 0
    if isinstance(wow_pct, float) and wow_pct < 10:
        # abuse_trend is a ratio, convert to percentage
        wow_pct = wow_pct * 100

    if pct > 4.0:
        suggestions.append({"priority": "CRITICAL",
            "message": f"%abuse at {pct:.2f}% exceeds 4% threshold. Identify top campaign by abuse amount. Check for campaign splitting. Deploy challenge on top abuser cluster immediately."})
    elif pct > 3.0:
        suggestions.append({"priority": "ALERT",
            "message": f"%abuse at {pct:.2f}% above 3% warning threshold. Investigate top campaign. Check Coin2DD path for SME involvement."})
    elif pct > 2.0:
        suggestions.append({"priority": "WATCH",
            "message": f"%abuse at {pct:.2f}%, above normal baseline (1.5–2.5%). Monitor for upward trend over next 2 weeks."})

    if pct < 1.5 and not detection_healthy:
        suggestions.append({"priority": "CAUTION",
            "message": f"%abuse at {pct:.2f}% is unusually low — this should NOT be read as low risk. FAD and/or BAD_V2 detection sources may be degraded. Run manual audit on top-abuse appIDs."})

    if wow_pct > 30:
        suggestions.append({"priority": "SPIKE",
            "message": f"Abuse increased +{wow_pct:.0f}% WoW. Likely new campaign or attack pattern. Check if new campaign was launched without Risk RA review."})

    if pct > 3.0:
        suggestions.append({"priority": "CHECK",
            "message": "Verify whether the top-abuse campaign was recently split into multiple campaignIDs. Aggregate manually by promotionName/scheme to get true abuse scope."})

    if not suggestions:
        suggestions.append({"priority": "STABLE",
            "message": f"%abuse at {pct:.2f}%, within normal baseline (1.5–2.5%). Continue BAU monitoring."})

    return suggestions


def analyze_coin2dd(records: List[Dict]) -> List[Dict[str, str]]:
    """Analyze coin2dd_monthly table. Compares latest month vs prior."""
    if not records:
        return [{"priority": "NOTE", "message": "No Coin2DD data available."}]

    curr = records[-1]
    suggestions = []

    pct = curr.get("pct_abuse", 0) or 0
    mom_change = curr.get("MoM_abuse_change", 0) or 0

    if pct > 7.0:
        suggestions.append({"priority": "CRITICAL",
            "message": f"Coin2DD abuse at {pct:.2f}% — SME merchants likely contributing via earn-side abuse. Investigate SME earn paths (FIRST_SCANQR, CHECK_CASHLOAN). Apply SME earn cap or task restriction."})
    elif pct > 5.0:
        suggestions.append({"priority": "ALERT",
            "message": f"Coin2DD abuse at {pct:.2f}% exceeds 5% signal threshold. Check SME merchants amplifying earn before Coin2DD cashout. Review top earn tasks by abuse rate."})
    elif pct > 4.0:
        suggestions.append({"priority": "WATCH",
            "message": f"Coin2DD abuse at {pct:.2f}%, trending above normal baseline (3–4%). Monitor weekly for sustained increase."})
    else:
        suggestions.append({"priority": "STABLE",
            "message": f"Coin2DD abuse at {pct:.2f}%, within baseline range (3–4%). Continue BAU monitoring."})

    if mom_change > 50_000_000:
        suggestions.append({"priority": "INVESTIGATE",
            "message": f"Coin2DD absolute abuse +{mom_change/1e6:.0f}M VND MoM. Investigate new abuser cohort or new earn mechanism."})

    return suggestions


def analyze_appid_breakdown(records: List[Dict]) -> List[Dict[str, str]]:
    """Analyze appid_fraud_breakdown table. Compares current vs prior month."""
    if not records:
        return [{"priority": "NOTE", "message": "No appID breakdown data available."}]

    import pandas as pd
    df = pd.DataFrame(records)
    months = sorted(df["report_month"].unique())
    if len(months) < 2:
        return [{"priority": "NOTE", "message": "Need at least 2 months of appID data."}]

    curr_m, prev_m = months[-1], months[-2]
    df_curr = df[df["report_month"] == curr_m]
    df_prev = df[df["report_month"] == prev_m]

    merged = df_curr.merge(df_prev[["appID", "fraud_loss_M"]],
                           on="appID", suffixes=("_curr", "_prev"), how="left")
    merged["fraud_loss_M_prev"] = merged["fraud_loss_M_prev"].fillna(0)
    merged["change_pct"] = (
        (merged["fraud_loss_M_curr"] - merged["fraud_loss_M_prev"])
        / merged["fraud_loss_M_prev"].replace(0, float("nan")) * 100
    ).fillna(0)

    suggestions = []

    # New high-volume threats
    new_high = merged[(merged["fraud_loss_M_curr"] > 50) & (merged["change_pct"] > 50)]
    for _, row in new_high.iterrows():
        suggestions.append({"priority": "ALERT",
            "message": f"NEW THREAT: appID {int(row['appID'])} ({row.get('app_name','')}) fraud +{row['change_pct']:.0f}% MoM to {row['fraud_loss_M_curr']:.0f}M VND. Deploy targeted rule immediately."})

    # Top concentration
    total_curr = df_curr["fraud_loss_M"].sum()
    if total_curr > 0:
        top = df_curr.nlargest(1, "fraud_loss_M").iloc[0]
        top_share = top["fraud_loss_M"] / total_curr * 100
        if top_share > 30:
            suggestions.append({"priority": "NOTE",
                "message": f"CONCENTRATION: appID {int(top['appID'])} ({top.get('app_name','')}) accounts for {top_share:.0f}% of total fraud. Further segmentation recommended."})

    # Effective controls
    effective = merged[(merged["change_pct"] < -40) & (merged["fraud_loss_M_prev"] > 100)]
    for _, row in effective.iterrows():
        suggestions.append({"priority": "CONFIRM",
            "message": f"CONTROL EFFECTIVE: appID {int(row['appID'])} ({row.get('app_name','')}) fraud {row['change_pct']:.0f}% MoM. Confirm rule is holding and check for fraud shift to adjacent appIDs."})

    # Top 3 by current loss
    top3 = df_curr.nlargest(3, "fraud_loss_M")
    top3_str = ", ".join(
        f"{int(r['appID'])} {r.get('app_name','')} ({r['fraud_loss_M']:.0f}M)"
        for _, r in top3.iterrows()
    )
    suggestions.append({"priority": "INFO",
        "message": f"Top 3 appIDs by fraud loss ({curr_m}): {top3_str}."})

    if not suggestions:
        suggestions.append({"priority": "STABLE",
            "message": f"No significant appID movement detected in {curr_m}."})

    return suggestions
