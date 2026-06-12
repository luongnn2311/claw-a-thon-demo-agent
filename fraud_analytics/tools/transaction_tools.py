from __future__ import annotations
from typing import Dict, Any
import numpy as np
from scipy import stats
from langchain_core.tools import tool
from fraud_analytics.utils.data_simulator import generate_mock_transactions


@tool
def query_transaction_summary(start_date: str, end_date: str) -> Dict[str, Any]:
    """Query overall transaction summary metrics for a date range.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Transaction summary with volume, amounts, success/failure rates, discount metrics.
    """
    df = generate_mock_transactions(start_date, end_date)
    total = len(df)
    success = int((df["transStatus"] == "SUCCESS").sum())
    failed = int((df["transStatus"] == "FAILED").sum())

    return {
        "period": f"{start_date} to {end_date}",
        "total_transactions": total,
        "total_amount": round(float(df["amount"].sum()), 2),
        "total_charged_amount": round(float(df["userchargeAmount"].sum()), 2),
        "total_discount_amount": round(float(df["discountAmount"].sum()), 2),
        "success_count": success,
        "failed_count": failed,
        "pending_count": int((df["transStatus"] == "PENDING").sum()),
        "success_rate_pct": round(success / total * 100, 2),
        "failure_rate_pct": round(failed / total * 100, 2),
        "avg_transaction_amount": round(float(df["amount"].mean()), 2),
        "median_transaction_amount": round(float(df["amount"].median()), 2),
        "discount_ratio_pct": round(
            float(df["discountAmount"].sum() / df["amount"].sum() * 100), 2
        ),
    }


@tool
def query_discount_analysis(start_date: str, end_date: str) -> Dict[str, Any]:
    """Analyze discount patterns and detect merchant-level discount abuse.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Discount analysis with per-merchant breakdown and anomaly flags.
    """
    df = generate_mock_transactions(start_date, end_date)

    merchant_discount = df.groupby("pmcID").agg(
        total_discount=("discountAmount", "sum"),
        total_amount=("amount", "sum"),
        txn_count=("transID", "count"),
    )
    merchant_discount["discount_ratio"] = (
        merchant_discount["total_discount"] / merchant_discount["total_amount"]
    )

    mean_r = float(merchant_discount["discount_ratio"].mean())
    std_r = float(merchant_discount["discount_ratio"].std())
    threshold = mean_r + 2 * std_r

    anomalous = merchant_discount[
        merchant_discount["discount_ratio"] > threshold
    ].index.tolist()

    top5 = merchant_discount.nlargest(5, "total_discount").reset_index()

    return {
        "total_discount_amount": round(float(df["discountAmount"].sum()), 2),
        "overall_discount_ratio_pct": round(
            float(df["discountAmount"].sum() / df["amount"].sum() * 100), 2
        ),
        "avg_merchant_discount_ratio_pct": round(mean_r * 100, 2),
        "anomaly_threshold_pct": round(threshold * 100, 2),
        "top_discount_merchants": [
            {
                "merchant_id": row["pmcID"],
                "total_discount": round(float(row["total_discount"]), 2),
                "discount_ratio_pct": round(float(row["discount_ratio"] * 100), 2),
            }
            for _, row in top5.iterrows()
        ],
        "anomalous_merchants": anomalous[:10],
        "anomalous_merchant_count": len(anomalous),
    }


@tool
def query_payment_solution_breakdown(start_date: str, end_date: str) -> Dict[str, Any]:
    """Break down transactions by payment solution and flag high-failure solutions.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Per-solution counts, amounts, and success/failure rates.
    """
    df = generate_mock_transactions(start_date, end_date)

    bd = df.groupby("paymentSolution").agg(
        count=("transID", "count"),
        total_amount=("amount", "sum"),
        success_count=("transStatus", lambda x: (x == "SUCCESS").sum()),
        failed_count=("transStatus", lambda x: (x == "FAILED").sum()),
    )
    bd["success_rate_pct"] = (bd["success_count"] / bd["count"] * 100).round(2)
    bd["failure_rate_pct"] = (bd["failed_count"] / bd["count"] * 100).round(2)
    bd["amount_share_pct"] = (bd["total_amount"] / bd["total_amount"].sum() * 100).round(2)

    high_failure = bd[bd["failure_rate_pct"] > 25].index.tolist()

    return {
        "breakdown": {
            sol: {
                "count": int(row["count"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "amount_share_pct": round(float(row["amount_share_pct"]), 2),
                "success_rate_pct": round(float(row["success_rate_pct"]), 2),
                "failure_rate_pct": round(float(row["failure_rate_pct"]), 2),
            }
            for sol, row in bd.iterrows()
        },
        "dominant_solution": str(bd["count"].idxmax()),
        "high_failure_rate_solutions": high_failure,
    }


@tool
def query_trend_comparison(
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> Dict[str, Any]:
    """Compare key transaction metrics between current and previous periods.

    Args:
        current_start: Current period start date (YYYY-MM-DD)
        current_end: Current period end date (YYYY-MM-DD)
        previous_start: Previous period start date (YYYY-MM-DD)
        previous_end: Previous period end date (YYYY-MM-DD)

    Returns:
        Side-by-side metrics comparison with percentage changes.
    """
    curr_df = generate_mock_transactions(current_start, current_end)
    prev_df = generate_mock_transactions(previous_start, previous_end)

    def _metrics(df) -> Dict[str, float]:
        total = len(df)
        success = (df["transStatus"] == "SUCCESS").sum()
        return {
            "total_transactions": total,
            "total_amount": round(float(df["amount"].sum()), 2),
            "success_rate_pct": round(float(success / total * 100), 2),
            "avg_amount": round(float(df["amount"].mean()), 2),
            "discount_ratio_pct": round(
                float(df["discountAmount"].sum() / df["amount"].sum() * 100), 2
            ),
        }

    def _pct_change(curr: float, prev: float) -> float:
        return round((curr - prev) / prev * 100, 2) if prev else 0.0

    curr = _metrics(curr_df)
    prev = _metrics(prev_df)

    return {
        "current_period": {"start": current_start, "end": current_end, **curr},
        "previous_period": {"start": previous_start, "end": previous_end, **prev},
        "changes": {
            "transaction_count_change_pct": _pct_change(
                curr["total_transactions"], prev["total_transactions"]
            ),
            "amount_change_pct": _pct_change(curr["total_amount"], prev["total_amount"]),
            "success_rate_change_ppt": round(
                curr["success_rate_pct"] - prev["success_rate_pct"], 2
            ),
            "discount_ratio_change_ppt": round(
                curr["discount_ratio_pct"] - prev["discount_ratio_pct"], 2
            ),
        },
    }


@tool
def query_daily_volume_anomalies(start_date: str, end_date: str) -> Dict[str, Any]:
    """Detect daily transaction volume anomalies using Z-score analysis.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Daily volume series with Z-scores and anomaly flags.
    """
    df = generate_mock_transactions(start_date, end_date)
    df["reqDate"] = df["reqDate"].apply(lambda d: d.strftime("%Y-%m-%d"))

    daily = (
        df.groupby("reqDate")
        .agg(txn_count=("transID", "count"), total_amount=("amount", "sum"))
        .reset_index()
    )

    z_scores = np.abs(stats.zscore(daily["txn_count"].values))
    daily["z_score"] = z_scores.round(3)
    daily["is_anomaly"] = daily["z_score"] > 2.5

    anomaly_days = daily[daily["is_anomaly"]]["reqDate"].tolist()

    return {
        "total_days": len(daily),
        "anomaly_days": anomaly_days,
        "anomaly_day_count": len(anomaly_days),
        "avg_daily_transactions": round(float(daily["txn_count"].mean()), 1),
        "max_daily_transactions": int(daily["txn_count"].max()),
        "min_daily_transactions": int(daily["txn_count"].min()),
        "daily_series": [
            {
                "date": row["reqDate"],
                "txn_count": int(row["txn_count"]),
                "total_amount": round(float(row["total_amount"]), 2),
                "z_score": round(float(row["z_score"]), 3),
                "is_anomaly": bool(row["is_anomaly"]),
            }
            for _, row in daily.iterrows()
        ],
    }
