"""CICIDS2017 data preprocessing pipeline.

This script does 5 things:
1) Load CICIDS2017 CSV file
2) Clean missing values
3) Encode text labels to numbers
4) Split data into train and test
5) Save processed data files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Read command-line arguments from terminal."""
    parser = argparse.ArgumentParser(description="Preprocess a CICIDS2017 CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to CICIDS2017 CSV file (example: data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder to save processed files (default: data/processed)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full preprocessing pipeline."""
    args = parse_args()

    input_csv = args.input
    output_dir = args.output_dir

    # Step 1: Load CICIDS2017 CSV file.
    # We first check if file exists to avoid confusing errors.
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # Remove extra spaces in column names (common in CICIDS CSV headers).
    df.columns = [c.strip() for c in df.columns]

    # Step 2: Clean bad numeric values.
    # Replace +inf/-inf with NaN so they can be removed.
    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # dropna() removes rows that contain empty/null values.
    rows_before = len(df)
    df = df.dropna().copy()
    print(f"Rows after removing missing/infinite values: {len(df)} (removed {rows_before - len(df)})")

    # Step 3: Encode labels.
    # CICIDS uses text labels like BENIGN, DoS, PortScan, etc.
    # We convert them to numbers so ML models can train.
    label_col = "Label"
    if label_col not in df.columns:
        raise KeyError(f"Column '{label_col}' not found in dataset.")

    unique_labels = sorted(df[label_col].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df["LabelEncoded"] = df[label_col].map(label_to_id)
    print(f"Encoded {len(unique_labels)} label classes.")

    # Step 4: Split data into train and test sets.
    # X = features, y = target label.
    X = df.drop(columns=[label_col, "LabelEncoded"])
    y = df["LabelEncoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows : {len(X_test)}")

    # Step 5: Save processed data files.
    # We save full cleaned data + train/test splits + label mapping.
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_dir / "cleaned_full.csv"
    train_file = output_dir / "train.csv"
    test_file = output_dir / "test.csv"
    mapping_file = output_dir / "label_mapping.csv"

    df.to_csv(cleaned_file, index=False)
    X_train.assign(LabelEncoded=y_train.values).to_csv(train_file, index=False)
    X_test.assign(LabelEncoded=y_test.values).to_csv(test_file, index=False)
    pd.Series(label_to_id, name="label_id").to_csv(mapping_file)

    print("Saved files:")
    print(f"- {cleaned_file}")
    print(f"- {train_file}")
    print(f"- {test_file}")
    print(f"- {mapping_file}")


if __name__ == "__main__":
    main()
