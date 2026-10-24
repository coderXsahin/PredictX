
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")

target = "Machine failure"
excluded = [
    "UDI", "Product ID", target,
    "TWF", "HDF", "PWF", "OSF", "RNF",
]

X = df.drop(columns=excluded)
y = df[target]

# Create training and held-out test partitions.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Split training data again to create a validation set.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    stratify=y_train_full,
    random_state=21,
)

numeric_columns = [
    c for c in X.columns if c != "Type"
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_columns,
    ),
    (
        "type",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        ["Type"],
    ),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_fit, y_fit)

probabilities = model.predict_proba(X_val)[:, 1]

print("Validation records:", len(X_val))
print(
    "Validation average precision:",
    round(average_precision_score(y_val, probabilities), 4),
)

results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms": fp,
        "missed_failures": fn,
        "detected_failures": tp,
    })

results_df = pd.DataFrame(results)

print("\n--- VALIDATION THRESHOLD COMPARISON ---")
print(results_df.round(3).to_string(index=False))

output = ROOT / "reports" / "threshold_comparison.csv"
results_df.to_csv(output, index=False)
print("\nSaved:", output)
