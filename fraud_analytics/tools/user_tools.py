from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import stats
from langchain_core.tools import tool
from fraud_analytics.utils.data_simulator import generate_mock_transactions


@tool
def query_user_metrics(start_date: str, end_date: str) -> Dict[str, Any]:
    """Analyze user-level transaction patterns and flag high-frequency / suspicious users.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        User activity metrics, new/repeat user breakdown, and suspicious user list.
    """
    df = generate_mock_transactions(start_date, end_date)

    user_stats = df.groupby("userID").agg(
        txn_count=("transID", "count"),
        total_amount=("amount", "sum"),
        unique_merchants=("pmcID", "nunique"),
        total_discount=("discountAmount", "sum"),
    )

    days = max(
        1,
        (
            __import__("datetime").datetime.strptime(end_date, "%Y-%m-%d")
            - __import__("datetime").datetime.strptime(start_date, "%Y-%m-%d")
        ).days + 1,
    )
    freq_threshold = max(10, days * 5)
    high_freq = user_stats[user_stats["txn_count"] > freq_threshold]

    user_ids_numeric = user_stats.index.str.extract(r"(\d+)")[0].astype(int).values
    new_users = user_stats[user_ids_numeric > 4500]
    repeat_users = user_stats[user_stats["txn_count"] >= 3]

    z_scores = np.abs(stats.zscore(user_stats["txn_count"].values))
    suspicious = user_stats.index[z_scores > 3.0].tolist()

    top5 = user_stats.nlargest(5, "txn_count").reset_index()

    return {
        "total_active_users": len(user_stats),
        "new_users_count": len(new_users),
        "repeat_users_count": len(repeat_users),
        "high_frequency_users_count": len(high_freq),
        "high_frequency_threshold": int(freq_threshold),
        "suspicious_users": suspicious[:10],
        "suspicious_user_count": len(suspicious),
        "avg_txn_per_user": round(float(user_stats["txn_count"].mean()), 2),
        "top_users_by_volume": [
            {
                "user_id": row["userID"],
                "txn_count": int(row["txn_count"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "unique_merchants": int(row["unique_merchants"]),
            }
            for _, row in top5.iterrows()
        ],
    }


@tool
def query_user_discount_behavior(start_date: str, end_date: str) -> Dict[str, Any]:
    """Identify users with abnormal discount usage — potential promotion abuse.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        User discount statistics, anomaly threshold, and top discount abusers.
    """
    df = generate_mock_transactions(start_date, end_date)

    ud = df.groupby("userID").agg(
        total_discount=("discountAmount", "sum"),
        total_amount=("amount", "sum"),
        txn_count=("transID", "count"),
    )
    ud["discount_ratio"] = ud["total_discount"] / ud["total_amount"].replace(0, float("nan"))

    mean_r = float(ud["discount_ratio"].mean())
    std_r = float(ud["discount_ratio"].std())
    threshold = mean_r + 2 * std_r

    high_discount = ud[ud["discount_ratio"] > threshold]
    top5 = high_discount.nlargest(5, "total_discount").reset_index()

    return {
        "avg_user_discount_ratio_pct": round(mean_r * 100, 2),
        "anomaly_threshold_pct": round(threshold * 100, 2),
        "high_discount_user_count": len(high_discount),
        "top_discount_abusers": [
            {
                "user_id": row["userID"],
                "total_discount": round(float(row["total_discount"]), 2),
                "discount_ratio_pct": round(float(row["discount_ratio"] * 100), 2),
                "txn_count": int(row["txn_count"]),
            }
            for _, row in top5.iterrows()
        ],
        "total_discount_from_abusers": round(float(high_discount["total_discount"].sum()), 2),
    }
