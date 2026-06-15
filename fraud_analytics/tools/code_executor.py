"""
Pandas code generator and executor for querying raw input CSV files.
- All input DataFrames are pre-loaded into the exec namespace.
- Results are capped at MAX_ROWS rows and returned as a markdown table.
"""
from __future__ import annotations
import os
import re
import pandas as pd
from typing import Dict, Tuple

_INPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../projectF/data input")
)

MAX_ROWS = 20

# variable_name → (filename, comma-separated column list)
_INPUT_FILES: Dict[str, Tuple[str, str]] = {
    "df_promo_txn": (
        "promo_transactions_with_abuse_flag.csv",
        "transID, campaignID, campaignCode, promotionName, userID, discountAmount, "
        "reqDate, transStatus, transType, appID, platform, amount, userChargeAmount, "
        "is_abuser, is_monitoring, is_malicious, is_acquittal, isEstAbuser, "
        "abuse_discount, month, week_start",
    ),
    "df_tpe": (
        "raw_clean_tpe_log.csv",
        "transID, userID, deviceID, appID, transType, transStatus, pmcID, "
        "paymentSolution, amount, userChargeAmount, discountAmount, bankCode, "
        "platform, reqDate, description, report_cat",
    ),
    "df_user_profile": (
        "raw_dim_user_promo_profile.csv",
        "userID, req_date, successful_dct_amt, successful_pmt_amount, "
        "successful_pmt_user_charge_amt, successful_pmt_user_charge_amt_wo_promo, "
        "successful_pmt_txns, successful_pmt_amt, successful_pmt_amt_wo_promo",
    ),
    "df_pom_wallet": (
        "raw_post_mortem_account_risk.csv",
        "transID, userID, deviceID, bimID, appID, pmcID, bankCode, amount, "
        "userChargeAmount, discountAmount, platform, transType, transStatus, "
        "paymentSolution, reqDate, report_date, first6CardNo, last4CardNo, "
        "first6Last4, is_kyc, map_type, integratedChannel, "
        "totalAmountSuccessWithin24h, totalTransSuccessWithin24h, source, fraud_type",
    ),
    "df_pom_gateway": (
        "raw_post_mortem_account_risk_gateway.csv",
        "transID, first6Last4, userID, appID, pmcID, bankCode, amount, "
        "userChargeAmount, transType, transStatus, paymentSolution, reqDate, "
        "report_date, first6CardNo, last4CardNo, bimID, integratedChannel, "
        "totalAmountSuccessWithin24h, totalTransSuccessWithin24h, source, fraud_type",
    ),
    "df_abuser": (
        "raw_promo_abuse_dim_user_v2.csv",
        "user_id, is_abuser, is_monitoring, is_malicious, is_acquittal",
    ),
    "df_promo_history": (
        "raw_promotion_transaction_history.csv",
        "transID, campaignID, campaignCode, promotionName, userID, discountAmount, "
        "reqDate, transStatus, transType, appID, platform, amount, userChargeAmount",
    ),
}

# Patterns that are never safe in generated code
_BLOCKED = re.compile(
    r"\b(import\s|__import__|subprocess|os\.system|open\s*\(|shutil|socket"
    r"|exec\s*\(|eval\s*\(|compile\s*\()\b",
    re.IGNORECASE,
)


def build_schema_context() -> str:
    """Return a schema description for each available input file."""
    lines = []
    for var, (fname, cols) in _INPUT_FILES.items():
        path = os.path.join(_INPUT_DIR, fname)
        if os.path.exists(path):
            lines.append(f"  {var}  ←  {fname}\n    Columns: {cols}")
    return "\n".join(lines) if lines else "  (no input files found)"


def _to_markdown(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    df = df.head(max_rows).reset_index(drop=True)
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + body)


def _load_namespace() -> Dict:
    ns: Dict = {"pd": pd}
    for var, (fname, _) in _INPUT_FILES.items():
        path = os.path.join(_INPUT_DIR, fname)
        if os.path.exists(path):
            try:
                ns[var] = pd.read_csv(path, low_memory=False)
            except Exception:
                pass
    return ns


def execute_pandas_code(code: str) -> str:
    """
    Execute LLM-generated pandas code in a pre-loaded namespace.

    Rules for the generated code:
    - All input DataFrames are already loaded — do NOT call pd.read_csv.
    - The final answer must be assigned to `result` (DataFrame or scalar).
    - DataFrame results are automatically capped at MAX_ROWS rows.
    """
    # Strip markdown fences if the LLM wrapped the code
    code = re.sub(r"^```[a-z]*\n?", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"\n?```$", "", code.strip())

    if _BLOCKED.search(code):
        return "Code contains disallowed operations and was not executed."

    ns = _load_namespace()
    try:
        exec(compile(code, "<query>", "exec"), ns)
    except Exception as exc:
        return f"Execution error: {exc}\n\nGenerated code:\n```python\n{code}\n```"

    result = ns.get("result")
    if result is None:
        return "Code executed but did not assign a `result` variable."

    if isinstance(result, pd.DataFrame):
        total = len(result)
        shown = min(total, MAX_ROWS)
        md = _to_markdown(result, shown)
        note = (
            f"\n\n*Showing {shown} of {total} rows.*"
            if total > shown
            else f"\n\n*{total} rows total.*"
        )
        return md + note

    if isinstance(result, pd.Series):
        df = result.reset_index()
        df.columns = [str(c) for c in df.columns]
        total = len(df)
        shown = min(total, MAX_ROWS)
        md = _to_markdown(df, shown)
        note = (
            f"\n\n*Showing {shown} of {total} rows.*"
            if total > shown
            else f"\n\n*{total} rows total.*"
        )
        return md + note

    return str(result)
