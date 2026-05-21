# To download the Amazon Electronics dataset:
# 1. pip install kaggle
# 2. Place your kaggle.json API token in ~/.kaggle/kaggle.json
#    (Get it from https://www.kaggle.com/settings -> "Create New Token")
# 3. Run: python scripts/download_dataset.py
#
# Alternatively (manual download):
# - Go to https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
# - Download and extract the CSV
# - Place amazon.csv in the data/ directory, then re-run this script

import os
import subprocess
import sys

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "amazon.csv")
OUT_CSV = os.path.join(DATA_DIR, "electronics_catalog.csv")

ELECTRONICS_KEYWORDS = ["Electronics", "Computers", "Smartphones", "Headphones", "Cameras", "Tablets"]


def download_via_kaggle():
    print("Downloading dataset via Kaggle API...")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "karkavelrajaj/amazon-sales-dataset",
            "-p", DATA_DIR,
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Kaggle download failed:\n{result.stderr}")
        print("Please download manually and place amazon.csv in the data/ directory.")
        sys.exit(1)
    print("Download complete.")


def filter_electronics():
    if not os.path.exists(RAW_CSV):
        print(f"Raw CSV not found at {RAW_CSV}.")
        print("Please download manually and place amazon.csv in the data/ directory.")
        sys.exit(1)

    df = pd.read_csv(RAW_CSV, on_bad_lines="skip")
    print(f"Loaded {len(df)} total rows.")

    pattern = "|".join(ELECTRONICS_KEYWORDS)
    mask = df["category"].astype(str).str.contains(pattern, case=False, na=False)
    filtered = df[mask].copy()
    print(f"Filtered to {len(filtered)} electronics rows.")

    filtered.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")
    return len(filtered)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_CSV):
        download_via_kaggle()

    count = filter_electronics()
    print(f"\nDone. {count} electronics products ready for ingestion.")
    print(f"Next step: python scripts/ingest.py")
