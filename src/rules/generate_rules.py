"""Generate simple defense rules from SHAP-important features.

Idea:
- Read top SHAP features from a CSV (feature + importance)
- For each important feature, estimate a threshold from processed data
- Convert each feature to a simple firewall-style rule text

Example output:
IF Flow Bytes/s > 12345.67 THEN block IP
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate simple defense rules from SHAP features.")
    parser.add_argument(
        "--importance-csv",
        type=Path,
        default=Path("data/processed/shap_feature_importance.csv"),
        help="CSV with SHAP importance columns: feature, importance",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/processed/cleaned_full.csv"),
        help="Processed data CSV used to compute thresholds",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top features to convert into rules",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile for threshold calculation (default: 95)",
    )
    parser.add_argument(
        "--output-rules",
        type=Path,
        default=Path("src/rules/generated_rules.txt"),
        help="Path to save generated rule text",
    )
    return parser.parse_args()


def validate_inputs(importance_csv: Path, data_csv: Path) -> None:
    """Check that required input files exist."""
    if not importance_csv.exists():
        raise FileNotFoundError(f"Importance CSV not found: {importance_csv}")
    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")


def build_rule(feature: str, threshold: float) -> str:
    """Convert a feature+threshold into a simple readable defense rule."""
    # Beginner-friendly template rule.
    return f"IF {feature} > {threshold:.4f} THEN block IP"


def main() -> None:
    """Generate simple firewall-style rules from top SHAP features."""
    args = parse_args()
    validate_inputs(args.importance_csv, args.data_csv)

    # Load SHAP importance file and processed network-flow data.
    importance_df = pd.read_csv(args.importance_csv)
    data_df = pd.read_csv(args.data_csv)

    # Normalize column names to avoid issues with whitespace.
    importance_df.columns = [c.strip() for c in importance_df.columns]
    data_df.columns = [c.strip() for c in data_df.columns]

    # Check the SHAP file has required columns.
    required_cols = {"feature", "importance"}
    if not required_cols.issubset(set(importance_df.columns)):
        raise KeyError("Importance CSV must contain columns: feature, importance")

    # Pick the most important K features.
    top_features = (
        importance_df.sort_values("importance", ascending=False)
        .head(args.top_k)["feature"]
        .tolist()
    )

    rules: list[str] = []
    skipped: list[str] = []

    for feature in top_features:
        # Skip features not present in processed data.
        if feature not in data_df.columns:
            skipped.append(feature)
            continue

        # Convert to numeric if possible (rules need numeric thresholds).
        numeric_series = pd.to_numeric(data_df[feature], errors="coerce").dropna()
        if numeric_series.empty:
            skipped.append(feature)
            continue

        # Use percentile as threshold (e.g., 95th percentile = unusual/high value).
        threshold = float(numeric_series.quantile(args.percentile / 100.0))
        rules.append(build_rule(feature=feature, threshold=threshold))

    # Save rules to file.
    args.output_rules.parent.mkdir(parents=True, exist_ok=True)
    with args.output_rules.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated defense rules from SHAP-important features\n")
        f.write(f"# Top-K: {args.top_k}, Percentile: {args.percentile}\n\n")
        for idx, rule in enumerate(rules, start=1):
            f.write(f"Rule {idx}: {rule}\n")

        if skipped:
            f.write("\n# Skipped features (missing/non-numeric):\n")
            for feature in skipped:
                f.write(f"# - {feature}\n")

    print(f"Generated {len(rules)} rules.")
    if skipped:
        print(f"Skipped {len(skipped)} features (missing/non-numeric).")
    print(f"Rules saved to: {args.output_rules}")


if __name__ == "__main__":
    main()
