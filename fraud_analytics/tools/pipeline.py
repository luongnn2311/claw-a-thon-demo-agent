"""
Pandas pipeline — mirrors pyspark_pipeline.py logic.
Reads from projectF/data input/ CSVs and produces the 5 output DataFrames.
"""
from __future__ import annotations
import os
import json
import pandas as pd
from typing import Dict, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "../../projectF")
_INPUT_DIR = os.path.join(_BASE, "data input")
_OUTPUT_DIR = os.path.join(_BASE, "table to output")

# ── Lookup maps ───────────────────────────────────────────────────────────────
SEGMENT_MAP: Dict[int, str] = {
    454: "domestic_direct", 4093: "domestic_direct", 12: "domestic_direct",
    4698: "domestic_direct", 4699: "domestic_direct", 15: "domestic_direct",
    9999: "international", 579: "international", 2391: "international",
    6699: "international", 149: "international", 3762: "international",
}

APPNAME_MAP: Dict[int, str] = {
    454: "Nạp tiền", 9999: "Apple", 579: "Google Play", 2391: "Thẻ giải trí",
    6699: "App Store", 4093: "VNPAY", 12: "Thẻ điện thoại",
    4698: "Mã thẻ Viettel", 4699: "Mã thẻ Vinaphone",
    149: "Mobile Payment", 3762: "VNGGameShop", 15: "Dịch Vụ",
}


def _read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(_INPUT_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df


def _month_period_end(month_str: str, today: pd.Timestamp) -> str:
    """Return the period end for a month label 'YYYY-MM'.
    Current month → today's date.  Past months → last day of that month."""
    y, m = int(month_str[:4]), int(month_str[5:7])
    if y == today.year and m == today.month:
        return today.strftime("%Y-%m-%d")
    # last day of month
    last = pd.Timestamp(year=y, month=m, day=1) + pd.offsets.MonthEnd(1)
    return last.strftime("%Y-%m-%d")


def run_pipeline(start_date: str | None = None, end_date: str | None = None) -> Dict[str, Any]:
    """
    Run the full pipeline and return a dict with 5 DataFrames serialised as
    JSON-compatible records plus a 'success' flag.

    Weekly periods  = complete ISO weeks (Mon–Sun); the current partial week
                      is excluded.
    Monthly periods = full calendar months; the current month is cut at today.

    Args:
        start_date: ISO date string YYYY-MM-DD (optional filter)
        end_date:   ISO date string YYYY-MM-DD (optional filter)
    """
    # ── Calendar anchors ─────────────────────────────────────────────────────
    today = pd.Timestamp.now().normalize()
    # Last complete Monday-to-Sunday week
    last_complete_sunday = today - pd.to_timedelta(today.dayofweek + 1, unit="D")
    last_complete_monday = last_complete_sunday - pd.to_timedelta(6, unit="D")

    # ── Load inputs ───────────────────────────────────────────────────────────
    df_pom_wallet   = _read_csv("raw_post_mortem_account_risk.csv")
    df_pom_gateway  = _read_csv("raw_post_mortem_account_risk_gateway.csv")
    df_tpe          = _read_csv("raw_clean_tpe_log.csv")
    df_abuser       = _read_csv("raw_promo_abuse_dim_user_v2.csv")
    # df_dim_user     = _read_csv("raw_dim_user_promo_profile.csv")  # available if needed

    results: Dict[str, Any] = {}

    try:
        # ── Normalise POM data ────────────────────────────────────────────────
        for df in (df_pom_wallet, df_pom_gateway):
            df["reqDate"] = pd.to_datetime(df["reqDate"], errors="coerce")

        pom_all = pd.concat(
            [df_pom_wallet[["transID", "appID", "amount", "reqDate", "source", "fraud_type"]],
             df_pom_gateway[["transID", "appID", "amount", "reqDate", "source", "fraud_type"]]],
            ignore_index=True,
        )
        pom_all["segment"] = pom_all["appID"].map(SEGMENT_MAP).fillna("others")
        pom_all["loss_M"]  = pom_all["amount"] / 1_000_000
        pom_all["month"]   = pom_all["reqDate"].dt.strftime("%Y-%m")
        # ISO-week Monday
        pom_all["week_start"] = pom_all["reqDate"] - pd.to_timedelta(
            pom_all["reqDate"].dt.dayofweek, unit="D"
        )
        pom_all["week_start"] = pom_all["week_start"].dt.strftime("%Y-%m-%d")

        if start_date:
            pom_all = pom_all[pom_all["reqDate"] >= start_date]
        if end_date:
            pom_all = pom_all[pom_all["reqDate"] <= end_date]

        # ── TABLE 1 — Fraud Monthly Loss ──────────────────────────────────────
        monthly_pivot = (
            pom_all.groupby(["month", "segment"])["loss_M"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["domestic_direct", "international", "domestic_napas", "wallet", "others"]:
            if col not in monthly_pivot.columns:
                monthly_pivot[col] = 0.0
        monthly_pivot = monthly_pivot.rename(columns={
            "domestic_direct": "domestic_direct_loss",
            "international":   "international_loss",
            "domestic_napas":  "domestic_napas_loss",
            "wallet":          "wallet_loss",
            "others":          "others_loss",
        })
        monthly_pivot["total_loss"] = (
            monthly_pivot[["domestic_direct_loss","international_loss",
                           "domestic_napas_loss","wallet_loss","others_loss"]].sum(axis=1)
        )
        monthly_pivot = monthly_pivot.sort_values("month")
        monthly_pivot["MoM_change_total"] = monthly_pivot["total_loss"].diff().fillna(0).round(2)
        monthly_pivot["MoM_pct_change"] = (
            monthly_pivot["total_loss"].pct_change().fillna(0).mul(100).round(1)
        )
        # Percentage share columns
        monthly_pivot["domestic_direct_pct"] = (
            monthly_pivot["domestic_direct_loss"] / monthly_pivot["total_loss"].replace(0, float("nan")) * 100
        ).round(1).fillna(0)
        monthly_pivot["international_pct"] = (
            monthly_pivot["international_loss"] / monthly_pivot["total_loss"].replace(0, float("nan")) * 100
        ).round(1).fillna(0)
        # Period bounds — current month ends today; past months end on their last day
        monthly_pivot["period_start"] = monthly_pivot["month"] + "-01"
        monthly_pivot["period_end"]   = monthly_pivot["month"].apply(
            lambda m: _month_period_end(m, today)
        )
        monthly_pivot["is_partial"] = monthly_pivot["month"].apply(
            lambda m: int(m == today.strftime("%Y-%m"))
        )
        results["fraud_monthly_loss"] = monthly_pivot.to_dict(orient="records")

        # ── TABLE 2 — Fraud Weekly Loss (complete Mon–Sun weeks only) ───────────
        # Exclude the current partial week
        pom_complete_weeks = pom_all[
            pd.to_datetime(pom_all["week_start"]) <= last_complete_monday
        ].copy()

        weekly_pivot = (
            pom_complete_weeks.groupby(["week_start", "segment"])["loss_M"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["domestic_direct", "international", "domestic_napas", "wallet", "others"]:
            if col not in weekly_pivot.columns:
                weekly_pivot[col] = 0.0
        weekly_pivot = weekly_pivot.rename(columns={
            "domestic_direct": "domestic_direct_loss",
            "international":   "international_loss",
            "domestic_napas":  "domestic_napas_loss",
            "wallet":          "wallet_loss",
            "others":          "others_loss",
        })
        weekly_pivot["total_loss"] = (
            weekly_pivot[["domestic_direct_loss","international_loss",
                          "domestic_napas_loss","wallet_loss","others_loss"]].sum(axis=1)
        )
        weekly_pivot = weekly_pivot.sort_values("week_start")
        weekly_pivot["WoW_change"] = weekly_pivot["total_loss"].diff().fillna(0).round(2)
        weekly_pivot["WoW_pct_change"] = (
            weekly_pivot["total_loss"].pct_change().fillna(0).mul(100).round(1)
        )
        # Add week_end (Sunday) for clarity
        weekly_pivot["week_end"] = (
            pd.to_datetime(weekly_pivot["week_start"]) + pd.to_timedelta(6, unit="D")
        ).dt.strftime("%Y-%m-%d")
        results["fraud_weekly_loss"] = weekly_pivot.to_dict(orient="records")

        # ── Normalise TPE + abuser join ────────────────────────────────────────
        df_tpe = df_tpe[df_tpe["transStatus"] == 1].copy()
        df_tpe["reqDate"] = pd.to_datetime(df_tpe["reqDate"], errors="coerce")
        df_tpe["month"]   = df_tpe["reqDate"].dt.strftime("%Y-%m")
        df_tpe["week_start"] = (
            df_tpe["reqDate"] - pd.to_timedelta(df_tpe["reqDate"].dt.dayofweek, unit="D")
        ).dt.strftime("%Y-%m-%d")

        # abuser tagging
        df_abuser = df_abuser.rename(columns={"user_id": "userID"})
        df_abuser["isEstAbuser"] = (
            (df_abuser["is_abuser"] == 1) | (df_abuser["is_monitoring"] == 1)
        ).astype(int)
        df_abuser["isAbuser"] = (
            (df_abuser["is_abuser"] == 1) & (~df_abuser["is_acquittal"].isin([1, 2]))
        ).astype(int)

        tpe_joined = df_tpe.merge(
            df_abuser[["userID", "isEstAbuser"]],
            on="userID", how="left"
        )
        tpe_joined["isEstAbuser"] = tpe_joined["isEstAbuser"].fillna(0).astype(int)
        tpe_joined["abuse_amt"] = tpe_joined["amount"] * tpe_joined["isEstAbuser"]

        if start_date:
            tpe_joined = tpe_joined[tpe_joined["reqDate"] >= start_date]
        if end_date:
            tpe_joined = tpe_joined[tpe_joined["reqDate"] <= end_date]

        # ── TABLE 3 — Promo Weekly Abuse (complete Mon–Sun weeks only) ──────────
        tpe_complete_weeks = tpe_joined[
            pd.to_datetime(tpe_joined["week_start"]) <= last_complete_monday
        ].copy()

        promo_weekly = (
            tpe_complete_weeks.groupby("week_start")
            .agg(
                total_spending=("amount", "sum"),
                total_abuse=("abuse_amt", "sum"),
                total_abuser_users=("userID", lambda x: (
                    tpe_complete_weeks.loc[x.index[tpe_complete_weeks.loc[x.index, "isEstAbuser"] == 1], "userID"].nunique()
                )),
            )
            .reset_index()
            .sort_values("week_start")
        )
        promo_weekly["pct_abuse"] = (
            promo_weekly["total_abuse"] / promo_weekly["total_spending"].replace(0, float("nan")) * 100
        ).round(2).fillna(0)
        promo_weekly["WoW_abuse_change"] = promo_weekly["total_abuse"].diff().fillna(0).astype(int)
        promo_weekly["abuse_trend"] = (
            promo_weekly["total_abuse"].pct_change().fillna(0).round(3)
        )
        promo_weekly["week_end"] = (
            pd.to_datetime(promo_weekly["week_start"]) + pd.to_timedelta(6, unit="D")
        ).dt.strftime("%Y-%m-%d")
        results["promo_weekly_abuse"] = promo_weekly.to_dict(orient="records")

        # ── TABLE 4 — Coin2DD Monthly ──────────────────────────────────────────
        disc_tpe = tpe_joined[tpe_joined["discountAmount"] > 0].copy()
        disc_tpe["disc_abuse"] = disc_tpe["discountAmount"] * disc_tpe["isEstAbuser"]

        coin2dd = (
            disc_tpe.groupby("month")
            .agg(
                total_discount=("discountAmount", "sum"),
                total_abuse=("disc_abuse", "sum"),
                total_abuser_users=("userID", lambda x: (
                    disc_tpe.loc[x.index[disc_tpe.loc[x.index, "isEstAbuser"] == 1], "userID"].nunique()
                )),
            )
            .reset_index()
            .sort_values("month")
        )
        coin2dd["pct_abuse"] = (
            coin2dd["total_abuse"] / coin2dd["total_discount"].replace(0, float("nan")) * 100
        ).round(2).fillna(0)
        coin2dd["MoM_abuse_change"] = coin2dd["total_abuse"].diff().fillna(0).astype(int)
        coin2dd["period_start"] = coin2dd["month"] + "-01"
        coin2dd["period_end"]   = coin2dd["month"].apply(
            lambda m: _month_period_end(m, today)
        )
        coin2dd["is_partial"] = coin2dd["month"].apply(
            lambda m: int(m == today.strftime("%Y-%m"))
        )
        results["coin2dd_monthly"] = coin2dd.to_dict(orient="records")

        # ── TABLE 5 — AppID Fraud Breakdown ────────────────────────────────────
        pom_all["app_name"] = pom_all["appID"].map(APPNAME_MAP).fillna("Other")
        pom_all["report_month"] = pom_all["reqDate"].dt.strftime("%Y-%m")

        appid_breakdown = (
            pom_all.groupby(["report_month", "appID", "app_name", "segment"])
            .agg(
                fraud_loss_M=("loss_M", "sum"),
                fraud_txn_count=("transID", "count"),
            )
            .reset_index()
            .sort_values(["report_month", "fraud_loss_M"], ascending=[True, False])
        )
        # 3-month trend columns (jan, feb, mar pivot)
        pivot_3m = (
            appid_breakdown.groupby(["appID", "app_name", "segment", "report_month"])["fraud_txn_count"]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )
        months = sorted(appid_breakdown["report_month"].unique())
        if len(months) >= 3:
            m1, m2, m3 = months[-3], months[-2], months[-1]
            pivot_3m = pivot_3m.rename(columns={
                m1: f"fraud_{m1.split('-')[1].lower()}",
                m2: f"fraud_{m2.split('-')[1].lower()}",
                m3: f"fraud_{m3.split('-')[1].lower()}",
            })
        results["appid_fraud_breakdown"] = appid_breakdown.to_dict(orient="records")

        results["success"] = True
        results["tables_computed"] = list(results.keys())

    except Exception as exc:
        results["success"] = False
        results["error"] = str(exc)

    return results
