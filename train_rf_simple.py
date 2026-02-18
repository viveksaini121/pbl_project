"""Train a simple Random Forest model on cleaned CICIDS data + SHAP explainability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Path to cleaned train and test files created by cicids_preprocess_simple.py
train_path = Path("data/processed/train.csv")
test_path = Path("data/processed/test.csv")

# Load cleaned train and test data.
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separate input features (X) and output label (y).
X_train = train_df.drop(columns=["LabelEncoded"])
y_train = train_df["LabelEncoded"]

X_test = test_df.drop(columns=["LabelEncoded"])
y_test = test_df["LabelEncoded"]

# Create Random Forest model.
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

# Train model using training data.
model.fit(X_train, y_train)

# Predict labels for test data.
y_pred = model.predict(X_test)

# Calculate metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# Print metrics.
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# -------- SHAP explainability --------
# Use a smaller sample for speed when calculating SHAP values.
max_shap_samples = 1000
X_shap = X_test.iloc[: min(max_shap_samples, len(X_test))].copy()

# Build SHAP TreeExplainer for Random Forest.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# Handle binary/multiclass output formats from different SHAP versions.
if isinstance(shap_values, list):
    # Old format: list of arrays [n_classes][n_samples, n_features]
    # Stack then average absolute SHAP across classes and samples.
    shap_array = np.stack(shap_values, axis=0)
    mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1))
else:
    # New format is usually ndarray, often [n_samples, n_features, n_classes]
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 2))
    else:
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

# Create feature-importance table and keep top 10.
feature_importance = pd.DataFrame(
    {
        "feature": X_shap.columns,
        "importance": mean_abs_shap,
    }
).sort_values("importance", ascending=False)

top10 = feature_importance.head(10).sort_values("importance", ascending=True)

# Save top-10 graph.
output_plot = Path("data/processed/shap_top10_features.png")
output_plot.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["importance"])
plt.xlabel("Mean |SHAP value|")
plt.title("Top 10 Important Features (SHAP)")
plt.tight_layout()
plt.savefig(output_plot, dpi=150)
plt.close()

print("SHAP top-10 feature graph saved at:", output_plot)
