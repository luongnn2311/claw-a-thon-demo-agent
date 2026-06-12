from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from langchain_core.tools import tool
from fraud_analytics.utils.data_simulator import generate_mock_transactions


@tool
def query_merchant_metrics(start_date: str, end_date: str) -> Dict[str, Any]:
    """Analyze merchant-level transaction metrics and identify outlier merchants.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Merchant statistics, top merchants, anomaly flags, and concentration metrics.
    """
    df = generate_mock_transactions(start_date, end_date)

    ms = df.groupby("pmcID").agg(
        txn_count=("transID", "count"),
        total_amount=("amount", "sum"),
        unique_users=("userID", "nunique"),
        success_rate=("transStatus", lambda x: round(float((x == "SUCCESS").mean() * 100), 2)),
        avg_amount=("amount", "mean"),
        total_discount=("discountAmount", "sum"),
    )

    z_scores = np.abs(stats.zscore(ms["txn_count"].values))
    anomalous = ms.index[z_scores > 2.5].tolist()

    total_vol = float(ms["total_amount"].sum())
    top5_vol = float(ms.nlargest(5, "total_amount")["total_amount"].sum())

    top10 = ms.nlargest(10, "txn_count").reset_index()

    return {
        "total_merchants": len(ms),
        "merchant_concentration_top5_pct": round(top5_vol / total_vol * 100, 2),
        "avg_merchant_txn_count": round(float(ms["txn_count"].mean()), 2),
        "txn_count_std": round(float(ms["txn_count"].std()), 2),
        "top_merchants": [
            {
                "merchant_id": row["pmcID"],
                "txn_count": int(row["txn_count"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "unique_users": int(row["unique_users"]),
                "success_rate_pct": row["success_rate"],
                "total_discount": round(float(row["total_discount"]), 2),
            }
            for _, row in top10.iterrows()
        ],
        "anomalous_merchants": anomalous[:10],
        "anomalous_merchant_count": len(anomalous),
    }


@tool
def query_merchant_new_vs_existing(
    start_date: str, end_date: str, lookback_days: int = 30
) -> Dict[str, Any]:
    """Compare new merchant activity versus established merchants.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        lookback_days: Days to look back to define 'existing' merchants (default 30)

    Returns:
        New vs existing merchant counts, volume shares, and high-velocity new merchants.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    lookback_start = (start - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    current_df = generate_mock_transactions(start_date, end_date)
    historical_df = generate_mock_transactions(lookback_start, start_date)

    existing = set(historical_df["pmcID"].unique())
    current_set = set(current_df["pmcID"].unique())
    new_merchants = current_set - existing

    new_mask = current_df["pmcID"].isin(new_merchants)
    new_df = current_df[new_mask]

    high_velocity = (
        new_df.groupby("pmcID")
        .size()
        .reset_index(name="count")
        .query("count > 50")["pmcID"]
        .tolist()
    )

    return {
        "total_active_merchants": len(current_set),
        "new_merchants_count": len(new_merchants),
        "existing_merchants_count": len(current_set - new_merchants),
        "new_merchant_txn_share_pct": round(float(new_mask.mean() * 100), 2),
        "new_merchant_amount_share_pct": round(
            float(new_df["amount"].sum() / current_df["amount"].sum() * 100), 2
        ),
        "sample_new_merchants": list(new_merchants)[:5],
        "high_velocity_new_merchants": high_velocity[:5],
    }
