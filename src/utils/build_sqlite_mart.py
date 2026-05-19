from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import load_config, resolve_path
from credit_risk.data import load_raw_credit_data, validate_raw_schema
from credit_risk.preprocessing import clean_credit_data


MONTH_MAP = [
    ("2005-09-01", "pay_0", "bill_amt1", "pay_amt1"),
    ("2005-08-01", "pay_2", "bill_amt2", "pay_amt2"),
    ("2005-07-01", "pay_3", "bill_amt3", "pay_amt3"),
    ("2005-06-01", "pay_4", "bill_amt4", "pay_amt4"),
    ("2005-05-01", "pay_5", "bill_amt5", "pay_amt5"),
    ("2005-04-01", "pay_6", "bill_amt6", "pay_amt6"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite analytics mart from the credit dataset.")
    parser.add_argument("--config", default="config/model_config.yaml")
    return parser.parse_args()


def build_customer_table(clean: pd.DataFrame) -> pd.DataFrame:
    return clean[
        [
            "customer_id",
            "limit_bal",
            "sex",
            "sex_label",
            "education",
            "education_label",
            "marriage",
            "marriage_label",
            "age",
            "default_next_month",
        ]
    ].copy()


def build_monthly_statement_table(clean: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for month_index, (statement_month, pay_col, bill_col, payment_col) in enumerate(MONTH_MAP, start=1):
        month = clean[["customer_id", pay_col, bill_col, payment_col]].copy()
        month = month.rename(
            columns={
                pay_col: "repayment_status",
                bill_col: "bill_amount",
                payment_col: "payment_amount",
            }
        )
        month["statement_month"] = statement_month
        month["recency_rank"] = month_index
        month["delinquent_flag"] = (month["repayment_status"] > 0).astype(int)
        frames.append(month)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_path = resolve_path(config["paths"]["raw_data"])
    sqlite_path = resolve_path(config["paths"]["sqlite_mart"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file not found at {raw_path}. Run `python scripts/download_data.py` first."
        )

    raw = load_raw_credit_data(raw_path)
    validate_raw_schema(raw)
    clean = clean_credit_data(raw)
    customers = build_customer_table(clean)
    statements = build_monthly_statement_table(clean)

    scores_path = resolve_path(config["paths"]["model_scores"])
    band_summary_path = resolve_path(config["paths"]["risk_band_summary"])
    stress_path = resolve_path(config["paths"]["stress_results"])

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as connection:
        customers.to_sql("customers", connection, index=False, if_exists="replace")
        statements.to_sql("monthly_statements", connection, index=False, if_exists="replace")
        if scores_path.exists():
            pd.read_csv(scores_path).to_sql("model_scores", connection, index=False, if_exists="replace")
        if band_summary_path.exists():
            pd.read_csv(band_summary_path).to_sql(
                "risk_band_summary", connection, index=False, if_exists="replace"
            )
        if stress_path.exists():
            pd.read_csv(stress_path).to_sql("stress_test_results", connection, index=False, if_exists="replace")

    print(f"SQLite mart written to {sqlite_path}")


if __name__ == "__main__":
    main()
