# Data

This project uses the UCI **Default of Credit Card Clients** dataset.

## Source

- Dataset page: https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients
- DOI: https://doi.org/10.24432/C55S3H
- License: CC BY 4.0
- CSV mirror used by `scripts/download_data.py`: https://huggingface.co/datasets/scikit-learn/credit-card-clients

## Folder Policy

- `raw/`: original downloaded source file.
- `interim/`: temporary cleaning artifacts.
- `processed/`: engineered features, model scores, stress results, SQLite mart.
- `external/`: optional macro, policy, or portfolio overlays.

Large data files are intentionally not committed. Recreate them with:

```bash
python scripts/download_data.py
python scripts/run_pipeline.py
python scripts/build_sqlite_mart.py
```

