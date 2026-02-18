"""Baseline intrusion detection model using Random Forest.

This script trains a simple Random Forest model on processed CICIDS2017 data.
It prints common evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 score
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline Random Forest IDS model.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train.csv"),
        help="Path to processed training CSV (default: data/processed/train.csv)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to processed testing CSV (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/baseline_rf.joblib"),
        help="Path to save trained model (default: models/baseline_rf.joblib)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the Random Forest (default: 200)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)


def main() -> None:
    """Train and evaluate the baseline model."""
    args = parse_args()

    # Load processed train and test files.
    train_df = load_dataset(args.train)
    test_df = load_dataset(args.test)

    # Check that target column exists.
    # LabelEncoded is the numeric label made during preprocessing.
    target_col = "LabelEncoded"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise KeyError(f"Expected target column '{target_col}' not found in train/test files.")

    # Split each dataset into features (X) and labels (y).
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Safety cleanup: replace inf values with NaN and remove bad rows.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    valid_train = ~X_train.isna().any(axis=1)
    valid_test = ~X_test.isna().any(axis=1)

    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    X_test = X_test.loc[valid_test]
    y_test = y_test.loc[valid_test]

    print(f"Training rows after cleanup: {len(X_train)}")
    print(f"Testing rows after cleanup : {len(X_test)}")

    # Build a baseline Random Forest model.
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )

    # Train the model.
    model.fit(X_train, y_train)

    # Save model so Streamlit app can use it.
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"Saved trained model to: {args.model_out}")

    # Predict labels for test data.
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics.
    # weighted average handles multiclass labels safely.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()
