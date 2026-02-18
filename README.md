# CICIDS Simple Preprocessing

This script does 5 steps:
1. Load one CICIDS CSV file
2. Remove missing values
3. Convert attack labels to numbers
4. Split into train and test
5. Save cleaned files

## How to check if you are doing everything correctly

Run this checklist in Terminal:

```bash
# 1) You must be inside your project folder
cd ~/xnids
pwd

# 2) These two files must exist in this same folder
ls -l requirements.txt cicids_preprocess_simple.py

# 3) Your dataset CSV files should be in data/
ls data/*.csv

# 4) Create/activate venv and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5) Run setup check only (no processing yet)
python cicids_preprocess_simple.py --check-only

# 6) If check looks good, run preprocessing
python cicids_preprocess_simple.py

# 7) Verify output files
ls -l data/processed/
```

## Why your earlier command failed
You executed commands where these files were missing:
- `requirements.txt`
- `cicids_preprocess_simple.py`

## Optional: use a specific CSV file

```bash
python cicids_preprocess_simple.py --input "data/<your-file-name>.csv"
```

## Output files
The script saves:
- `data/processed/cleaned_full.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/label_mapping.csv`
