# CICIDS Simple Preprocessing

This script does 5 steps:
1. Load one CICIDS CSV file
2. Remove missing values
3. Convert attack labels to numbers
4. Split into train and test
5. Save cleaned files

## How to check if each step is done or not

Run these commands:

```bash
cd ~/xnids
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cicids_preprocess_simple.py --check-only
python cicids_preprocess_simple.py --input "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
ls -l data/processed/
```

When processing runs, the script prints:
- `✅ Step 1 done ...`
- `✅ Step 2 done ...`
- `✅ Step 3 done ...`
- `✅ Step 4 done ...`
- `✅ Step 5 done ...`

If a step fails, Python stops and shows an error.

## Simple Python code

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1) Load CICIDS CSV
df = pd.read_csv("data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# 2) Remove missing values
df = df.dropna()

# 3) Convert attack labels to numbers
df.columns = [c.strip() for c in df.columns]
label_to_id = {label: i for i, label in enumerate(sorted(df["Label"].unique()))}
df["LabelEncoded"] = df["Label"].map(label_to_id)

# 4) Split into train and test
X = df.drop(columns=["Label", "LabelEncoded"])
y = df["LabelEncoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Save cleaned data
df.to_csv("data/processed/cleaned_full.csv", index=False)
X_train.assign(LabelEncoded=y_train.values).to_csv("data/processed/train.csv", index=False)
X_test.assign(LabelEncoded=y_test.values).to_csv("data/processed/test.csv", index=False)
```

## Explain every line in simple words

- `import pandas as pd`: load pandas library to work with tables.
- `from sklearn.model_selection import train_test_split`: load helper to split train/test.
- `df = pd.read_csv(...)`: read CSV file into memory.
- `df = df.dropna()`: remove rows that have missing values.
- `df.columns = [c.strip() ...]`: remove extra spaces from column names.
- `label_to_id = {...}`: create mapping from label text to number.
- `df["LabelEncoded"] = ...`: create numeric target column.
- `X = df.drop(...)`: features without target columns.
- `y = df["LabelEncoded"]`: target column.
- `train_test_split(...)`: split data into training and testing parts.
- `df.to_csv(...)`: save full cleaned dataset.
- `X_train.assign(...).to_csv(...)`: save train set with target.
- `X_test.assign(...).to_csv(...)`: save test set with target.
