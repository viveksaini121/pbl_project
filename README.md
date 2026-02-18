# X-NIDS

X-NIDS is an Explainable AI-based Intrusion Detection System project focused on detecting network attacks while providing transparent, human-understandable model explanations and actionable security rules.

## Streamlit app (upload CSV + attack prediction + SHAP)

Run the app:

```bash
streamlit run app/app.py
```

### What this app does
- Uploads a network CSV file.
- Uses trained model at `models/baseline_rf.joblib`.
- Predicts attack class for each row.
- Shows SHAP explanation for first row.
- Shows top 10 SHAP features in a table.

### Explain every line (simple)
- `from pathlib import Path`: helps work with file paths safely.
- `import joblib`: loads saved ML model file.
- `import pandas as pd`: reads CSV data.
- `import shap`: computes explainability values.
- `import streamlit as st`: builds web UI.
- `st.title(...)`: app heading.
- `st.write(...)`: short instruction text.
- `MODEL_PATH ...`: where model file is expected.
- `LABEL_MAP_PATH ...`: optional label mapping file path.
- `if not MODEL_PATH.exists()`: stop app if model is missing.
- `model = joblib.load(...)`: load trained model into memory.
- `uploaded_file = st.file_uploader(...)`: user uploads CSV.
- `pd.read_csv(uploaded_file)`: read uploaded file.
- `input_df.columns = ...strip()`: clean header spaces.
- `drop Label/LabelEncoded`: remove target columns from input.
- `model.feature_names_in_`: required feature order from training.
- `missing features -> 0`: fill absent columns safely.
- `X = input_df[expected_features]`: align columns exactly.
- `model.predict(X)`: generate predictions.
- `id_to_label`: convert numeric class to readable class (if mapping exists).
- `shap.TreeExplainer(model)`: SHAP explainer for tree model.
- `shap_values ...`: compute feature contribution values.
- `waterfall(...)`: visualize top feature effects for first row.
- `abs(sv) ... head(10)`: top 10 strongest SHAP features table.
