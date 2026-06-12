from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _date_seed(start_date: str, end_date: str) -> int:
    return abs(hash(start_date + end_date)) % (2**32)


def generate_mock_transactions(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate deterministic mock transaction data for a date range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days + 1

    seed = _date_seed(start_date, end_date)
    rng = np.random.default_rng(seed)

    n = days * 1000

    statuses = rng.choice(
        ["SUCCESS", "FAILED", "PENDING"], size=n, p=[0.78, 0.17, 0.05]
    )
    amounts = rng.lognormal(mean=10.5, sigma=1.2, size=n).round(2)
    discount_ratios = rng.beta(a=1, b=8, size=n)

    payment_solutions = rng.choice(
        ["VISA", "MASTERCARD", "VNPAY", "MOMO", "ZALOPAY", "BANK_TRANSFER"],
        size=n,
        p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08],
    )

    merchants = [f"PMC_{i:04d}" for i in range(1, 201)]
    merchant_weights = np.exp(-np.arange(200) * 0.05)
    merchant_weights /= merchant_weights.sum()

    day_offsets = rng.integers(0, days, size=n)
    dates = [start + timedelta(days=int(d)) for d in day_offsets]

    return pd.DataFrame(
        {
            "transID": [f"TXN_{i:08d}" for i in range(n)],
            "appID": rng.choice(["APP_001", "APP_002", "APP_003"], size=n),
            "userID": [f"USR_{rng.integers(1, 5001):06d}" for _ in range(n)],
            "pmcID": rng.choice(merchants, size=n, p=merchant_weights),
            "paymentSolution": payment_solutions,
            "amount": amounts,
            "userchargeAmount": (amounts * (1 - discount_ratios)).round(2),
            "discountAmount": (amounts * discount_ratios).round(2),
            "transType": rng.choice(
                ["PAYMENT", "REFUND", "TOP_UP"], size=n, p=[0.85, 0.10, 0.05]
            ),
            "transStatus": statuses,
            "reqDate": dates,
        }
    )
