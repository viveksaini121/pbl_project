"""Simple CICIDS CSV preprocessing script.

Steps:
1) Load one CICIDS CSV
2) Remove missing values
3) Convert attack labels to numbers
4) Split into train and test
5) Save cleaned data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess one CICIDS CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to a CICIDS CSV file. If omitted, script tries data/*.csv then xnids/data/*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Folder where cleaned files will be written (default: data/processed)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify paths and print setup status; do not process data.",
    )
    return parser.parse_args()


def resolve_input_path(input_arg: Path | None) -> Path:
    """Find input CSV from --input or common folders."""
    if input_arg is not None:
        return input_arg

    # Most users run from ~/xnids, so try data/*.csv first.
    candidates = sorted(Path("data").glob("*.csv"))
    if candidates:
        print(f"No --input provided. Using: {candidates[0]}")
        return candidates[0]

    # Fallback for repository layout where CSVs are under xnids/data.
    candidates = sorted(Path("xnids/data").glob("*.csv"))
    if candidates:
        print(f"No --input provided. Using: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "No CSV found automatically.\n"
        "Put a CSV in data/ or xnids/data/, or pass a path with --input"
    )


def run_setup_check(input_csv: Path, output_dir: Path) -> None:
    """Print a simple setup checklist."""
    print("Setup check:")
    print(f"- Working directory: {Path.cwd()}")
    print(f"- Input CSV path: {input_csv}")
    print(f"- Input exists: {input_csv.exists()}")
    print(f"- Output directory (will be used): {output_dir}")

    data_csvs = sorted(Path("data").glob("*.csv"))
    nested_csvs = sorted(Path("xnids/data").glob("*.csv"))
    print(f"- CSV count in data/: {len(data_csvs)}")
    print(f"- CSV count in xnids/data/: {len(nested_csvs)}")


def main() -> None:
    """Run all preprocessing steps."""
    args = parse_args()

    input_csv = resolve_input_path(args.input)
    output_dir = args.output_dir

    run_setup_check(input_csv=input_csv, output_dir=output_dir)
    if args.check_only:
        return

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Tip: pass the real file path using --input"
        )

    # Create output folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load CSV
    df = pd.read_csv(input_csv)

    # 2) Remove missing values
    df = df.dropna()

    # Normalize column names (CICIDS files can include extra spaces).
    df.columns = [col.strip() for col in df.columns]

    # 3) Convert attack labels to numbers
    label_col = "Label"
    if label_col not in df.columns:
        raise KeyError(
            f"Column '{label_col}' not found in {input_csv}. Available columns: {list(df.columns)[:10]}..."
        )

    unique_labels = sorted(df[label_col].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df["LabelEncoded"] = df[label_col].map(label_to_id)

    # 4) Split into train and test
    X = df.drop(columns=[label_col, "LabelEncoded"])
    y = df["LabelEncoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5) Save cleaned data
    df.to_csv(output_dir / "cleaned_full.csv", index=False)
    X_train.assign(LabelEncoded=y_train.values).to_csv(output_dir / "train.csv", index=False)
    X_test.assign(LabelEncoded=y_test.values).to_csv(output_dir / "test.csv", index=False)
    pd.Series(label_to_id, name="label_id").to_csv(output_dir / "label_mapping.csv")

    print("Done. Files saved in:", output_dir)
    print("Label mapping:", label_to_id)


if __name__ == "__main__":
    main()
