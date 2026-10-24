
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET = "Machine failure"

# Load data
df = pd.read_csv(DATA_PATH)

# Remove identifiers and failure-mode labels.
# The failure-mode columns can reveal the target and must not be inputs.
excluded_columns = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

X = df.drop(columns=excluded_columns)
y = df[TARGET]

categorical_features = ["Type"]
numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

# Split before fitting any preprocessing or model.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    result = {
        "model": name,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    results.append(result)
    trained_models[name] = pipeline

    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Failure"],
            zero_division=0,
        )
    )

results_df = pd.DataFrame(results).sort_values(
    by="average_precision",
    ascending=False,
)

results_path = REPORTS_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4).to_string(index=False))

# Select using average precision on the held-out comparison set.
# We'll later tune and validate the decision threshold properly.
best_name = results_df.iloc[0]["model"]
best_model = trained_models[best_name]

model_path = MODELS_DIR / "predictx_model.joblib"
joblib.dump(best_model, model_path)

print(f"\nSelected model: {best_name}")
print(f"Saved model: {model_path}")
print(f"Saved metrics: {results_path}")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
