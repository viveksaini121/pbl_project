"""Streamlit app for X-NIDS prediction + SHAP explanation.

This app lets a user:
1) Upload a network-flow CSV file
2) Run attack prediction using a trained model
3) View SHAP-based feature explanation
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

# Title shown at the top of the web app.
st.title("X-NIDS: Attack Prediction with Explainable AI")

# Small intro text for beginner users.
st.write(
    "Upload a network CSV file, run attack prediction, and view SHAP explanation for why the model predicted that class."
)

# Default paths for model and optional label mapping.
MODEL_PATH = Path("models/baseline_rf.joblib")
LABEL_MAP_PATH = Path("data/processed/label_mapping.csv")

# Show model path and check if file exists.
st.caption(f"Model path: {MODEL_PATH}")
if not MODEL_PATH.exists():
    st.error(
        "Trained model file not found at models/baseline_rf.joblib. "
        "Please train and save your model first."
    )
    st.stop()

# Load trained model once and reuse it.
model = joblib.load(MODEL_PATH)

# Try loading label mapping (optional helper for readable class names).
id_to_label: dict[int, str] = {}
if LABEL_MAP_PATH.exists():
    label_map_df = pd.read_csv(LABEL_MAP_PATH)
    # Handles mapping created by pandas Series.to_csv (first column index + label_id column).
    if label_map_df.shape[1] >= 2:
        # First column is usually label text; second is numeric id.
        for _, row in label_map_df.iterrows():
            try:
                label_text = str(row.iloc[0])
                label_id = int(row.iloc[1])
                id_to_label[label_id] = label_text
            except Exception:
                # Skip malformed rows quietly.
                pass

# Upload widget for user CSV.
uploaded_file = st.file_uploader("Upload network CSV", type=["csv"])

# Stop here until user uploads a file.
if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

# Read uploaded CSV into DataFrame.
input_df = pd.read_csv(uploaded_file)

# Clean column names (remove accidental spaces).
input_df.columns = [c.strip() for c in input_df.columns]

# Remove target columns if present, because model expects only features.
for maybe_target in ["Label", "LabelEncoded"]:
    if maybe_target in input_df.columns:
        input_df = input_df.drop(columns=[maybe_target])

st.subheader("Uploaded data preview")
st.dataframe(input_df.head(5))

# Align uploaded features with model's expected feature names.
if not hasattr(model, "feature_names_in_"):
    st.error("Model does not include feature names. Re-train with pandas DataFrame input.")
    st.stop()

expected_features = list(model.feature_names_in_)

# Add missing columns with zero value.
missing_features = [f for f in expected_features if f not in input_df.columns]
for feature in missing_features:
    input_df[feature] = 0

# Keep only expected columns in exact order.
X = input_df[expected_features]

# Button to run prediction.
if st.button("Predict attack"):
    # Model predicts numeric class ids.
    y_pred = model.predict(X)

    # Convert numeric ids to text labels when mapping is available.
    if id_to_label:
        pred_labels = [id_to_label.get(int(v), str(v)) for v in y_pred]
    else:
        pred_labels = [str(v) for v in y_pred]

    # Show predictions.
    st.subheader("Predictions")
    result_df = pd.DataFrame({"PredictedClass": pred_labels})
    st.dataframe(result_df)

    # SHAP explanation for first row only (simple + fast).
    st.subheader("SHAP explanation (first row)")
    st.write("This chart shows which features pushed the prediction up or down for the first row.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.iloc[[0]])

    # Support old/new SHAP output format.
    if isinstance(shap_values, list):
        # Multi-class old format: choose predicted class index for first row.
        class_id = int(y_pred[0])
        class_id = max(0, min(class_id, len(shap_values) - 1))
        sv = shap_values[class_id][0]
        base_value = explainer.expected_value[class_id]
    else:
        # New format often [samples, features, classes] or [samples, features].
        if getattr(shap_values, "ndim", 0) == 3:
            class_id = int(y_pred[0])
            class_id = max(0, min(class_id, shap_values.shape[2] - 1))
            sv = shap_values[0, :, class_id]
            base_value = explainer.expected_value[class_id]
        else:
            sv = shap_values[0]
            base_value = explainer.expected_value

    # Build Explanation object and render waterfall chart in Streamlit.
    explanation = shap.Explanation(values=sv, base_values=base_value, data=X.iloc[0], feature_names=X.columns)
    fig = shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig.figure)

    # Also show top 10 absolute SHAP features as a simple table.
    abs_values = pd.Series(abs(sv), index=X.columns).sort_values(ascending=False).head(10)
    st.subheader("Top 10 important features (absolute SHAP)")
    st.dataframe(abs_values.rename("importance").reset_index().rename(columns={"index": "feature"}))
