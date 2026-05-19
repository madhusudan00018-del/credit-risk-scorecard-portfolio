from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import resolve_path
from credit_risk.data import DEFAULT_DATA_URL, download_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the UCI credit card default dataset.")
    parser.add_argument("--url", default=DEFAULT_DATA_URL, help="CSV URL to download.")
    parser.add_argument(
        "--output",
        default="data/raw/UCI_Credit_Card.csv",
        help="Output path for the raw CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = resolve_path(Path(args.output))
    download_dataset(output, args.url)
    print(f"Downloaded dataset to {output}")


if __name__ == "__main__":
    main()
