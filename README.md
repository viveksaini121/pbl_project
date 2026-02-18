# CICIDS Simple Preprocessing

Use `cicids_preprocess_simple.py` to:
- load one CICIDS CSV file,
- drop rows with missing values,
- encode attack labels to numbers,
- split into train/test,
- save cleaned outputs in `xnids/data/processed`.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python cicids_preprocess_simple.py
```
