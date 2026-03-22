import re
import math
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)

from feature_utils import TextExtractor, HandcraftedTransformer

# -----------------------------
# Load dataset (JSONL)
df = pd.read_json("data/problems.jsonl", lines=True)

# -----------------------------
# Targets
y_class = df["problem_class"]
y_score = df["problem_score"]

X = df[
    [
        "title",
        "description",
        "input_description",
        "output_description",
        "sample_io"
    ]
]
x

# -----------------------------
# Feature pipelines
text_pipeline = Pipeline([
    ("extract", TextExtractor()),
    ("tfidf", TfidfVectorizer(max_features=3000))
])

numeric_pipeline = Pipeline([
    ("extract", HandcraftedTransformer()),
    ("scale", StandardScaler())
])

features = FeatureUnion([
    ("text", text_pipeline),
    ("numeric", numeric_pipeline)
])

# -----------------------------
# Train / test split
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_score, test_size=0.2, random_state=42
)

# -----------------------------
# Models
clf_model = Pipeline([
    ("features", features),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

reg_model = Pipeline([
    ("features", features),
    ("reg", RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

# -----------------------------
# Train
clf_model.fit(X_train, y_train_cls)
reg_model.fit(X_train, y_train_reg)

# -----------------------------
# Save models
joblib.dump(clf_model, "models/autojudge_classifier.joblib")
joblib.dump(reg_model, "models/autojudge_regressor.joblib")

# =============================
# EVALUATION METRICS
print("\n========== MODEL EVALUATION ==========")

# Classification
y_pred_cls = clf_model.predict(X_test)
acc = accuracy_score(y_test_cls, y_pred_cls)
cm = confusion_matrix(y_test_cls, y_pred_cls)

print("Classification Accuracy:", acc)
print("Confusion Matrix:\n", cm)

# Regression
y_pred_reg = reg_model.predict(X_test)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

print("Regression MAE:", mae)
print("Regression RMSE:", rmse)
print("=====================================\n")
