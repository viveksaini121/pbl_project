"""Simple CICIDS CSV preprocessing script.

Steps:
1) Load one CICIDS CSV
2) Remove missing values
3) Convert attack labels to numbers
4) Split into train/test
5) Save cleaned outputs
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Path to your input CSV file.
input_csv = Path("xnids/data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# Folder where cleaned files will be saved.
output_dir = Path("xnids/data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Read the CICIDS CSV into a pandas DataFrame.
df = pd.read_csv(input_csv)

# Remove rows that contain any missing value.
df = df.dropna()

# Remove extra spaces from column names (CICIDS files often have them).
df.columns = [col.strip() for col in df.columns]

# Label column name used in CICIDS2017.
label_col = "Label"

# Build a mapping like {'BENIGN': 0, 'DoS': 1, ...}.
unique_labels = sorted(df[label_col].unique())
label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

# Convert text labels to numeric labels.
df["LabelEncoded"] = df[label_col].map(label_to_id)

# Split into features (X) and target (y).
X = df.drop(columns=[label_col, "LabelEncoded"])
y = df["LabelEncoded"]

# Split data into train and test sets (80% train, 20% test).
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Save fully cleaned data and split datasets.
df.to_csv(output_dir / "cleaned_full.csv", index=False)
X_train.assign(LabelEncoded=y_train.values).to_csv(output_dir / "train.csv", index=False)
X_test.assign(LabelEncoded=y_test.values).to_csv(output_dir / "test.csv", index=False)

# Save label map so you can decode numeric predictions later.
pd.Series(label_to_id, name="label_id").to_csv(output_dir / "label_mapping.csv")

print("Done. Files saved in:", output_dir)
print("Label mapping:", label_to_id)
