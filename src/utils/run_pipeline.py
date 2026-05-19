from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full credit risk modeling pipeline.")
    parser.add_argument(
        "--config",
        default="config/model_config.yaml",
        help="Path to model configuration YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(args.config)
    metrics = result["metrics"]
    metadata = result["metadata"]
    print("Pipeline completed")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average precision: {metrics['average_precision']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Optimal threshold: {metrics['threshold']:.4f}")


if __name__ == "__main__":
    main()
