"""Train a simple Random Forest model on cleaned CICIDS data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
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
