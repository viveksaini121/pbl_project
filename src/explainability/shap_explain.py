"""Explain a trained intrusion detection model using SHAP.

This script:
1) Loads a trained model file
2) Loads processed feature data
3) Computes SHAP values
4) Shows the top 10 most important features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def parse_args() -> argparse.Namespace:
    """Read command-line arguments from terminal."""
    parser = argparse.ArgumentParser(description="SHAP explainability for trained IDS model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/baseline_rf.joblib"),
        help="Path to trained model file (default: models/baseline_rf.joblib)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/test.csv"),
        help="Path to processed data for explanation (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top features to show (default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum rows to use for SHAP (default: 1000)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("data/processed/shap_top10_features.png"),
        help="Where to save top-feature plot (default: data/processed/shap_top10_features.png)",
    )
    return parser.parse_args()


def load_inputs(model_path: Path, data_path: Path) -> tuple[object, pd.DataFrame]:
    """Load model and data files, with friendly errors if missing."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df


def compute_mean_abs_shap(model: object, features: pd.DataFrame) -> np.ndarray:
    """Compute mean absolute SHAP value per feature.

    We support both old and new SHAP return formats.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Old SHAP format: list of arrays, one array per class.
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=0)  # [classes, samples, features]
        return np.mean(np.abs(shap_array), axis=(0, 1))

    # New SHAP format often returns ndarray.
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        # [samples, features, classes] -> average over samples and classes.
        return np.mean(np.abs(shap_array), axis=(0, 2))
    # [samples, features] -> average over samples.
    return np.mean(np.abs(shap_array), axis=0)


def main() -> None:
    """Run SHAP explanation flow."""
    args = parse_args()

    # Load trained model and processed data.
    model, df = load_inputs(args.model, args.data)

    # If label column exists, remove it because SHAP needs only input features.
    target_col = "LabelEncoded"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df

    # Limit rows for faster SHAP on large datasets.
    X_small = X.iloc[: min(args.max_samples, len(X))].copy()

    # Compute mean absolute SHAP importance per feature.
    mean_abs_shap = compute_mean_abs_shap(model=model, features=X_small)

    # Build a sorted table of feature importance values.
    importance_df = pd.DataFrame(
        {
            "feature": X_small.columns,
            "importance": mean_abs_shap,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(args.top_k)
    print("Top important features (SHAP):")
    print(top_features.to_string(index=False))

    # Plot top-k features as a horizontal bar chart.
    # We reverse order for nicer top-to-bottom display.
    top_plot = top_features.sort_values("importance", ascending=True)

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_plot["feature"], top_plot["importance"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {args.top_k} Important Features (SHAP)")
    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150)
    plt.close()

    print(f"Saved SHAP feature plot to: {args.output_plot}")


if __name__ == "__main__":
    main()
