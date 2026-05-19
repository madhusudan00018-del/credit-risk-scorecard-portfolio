from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


DEFAULT_DATA_URL = (
    "https://huggingface.co/datasets/scikit-learn/credit-card-clients/"
    "resolve/main/UCI_Credit_Card.csv"
)


COLUMN_RENAME_MAP = {
    "ID": "customer_id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default.payment.next.month": "default_next_month",
    "default payment next month": "default_next_month",
    "Y": "default_next_month",
}


def download_dataset(output_path: str | Path, url: str = DEFAULT_DATA_URL) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output.write_bytes(response.content)
    return output


def load_raw_credit_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        frame = pd.read_excel(path, header=1)
    else:
        frame = pd.read_csv(path)
    return standardize_credit_columns(frame)


def standardize_credit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.rename(columns=COLUMN_RENAME_MAP)
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.lower()
    )
    if "default_payment_next_month" in cleaned.columns:
        cleaned = cleaned.rename(columns={"default_payment_next_month": "default_next_month"})
    return cleaned


def validate_raw_schema(frame: pd.DataFrame) -> None:
    required = {
        "customer_id",
        "limit_bal",
        "sex",
        "education",
        "marriage",
        "age",
        "pay_0",
        "pay_2",
        "pay_3",
        "pay_4",
        "pay_5",
        "pay_6",
        "bill_amt1",
        "bill_amt2",
        "bill_amt3",
        "bill_amt4",
        "bill_amt5",
        "bill_amt6",
        "pay_amt1",
        "pay_amt2",
        "pay_amt3",
        "pay_amt4",
        "pay_amt5",
        "pay_amt6",
        "default_next_month",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_train_validation_test(
    frame: pd.DataFrame,
    target: str,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_valid, test = train_test_split(
        frame,
        test_size=test_size,
        stratify=frame[target],
        random_state=random_state,
    )
    validation_fraction = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_valid,
        test_size=validation_fraction,
        stratify=train_valid[target],
        random_state=random_state,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

