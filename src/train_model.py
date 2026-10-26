
from pathlib import Path
import json

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


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

TARGET = "Machine failure"
RANDOM_STATE = 42

# Threshold candidates are selected using validation data only.
THRESHOLDS = [
    round(value, 2)
    for value in [0.10, 0.15, 0.20, 0.25, 0.30,
                  0.35, 0.40, 0.45, 0.50, 0.55,
                  0.60, 0.65, 0.70, 0.75, 0.80,
                  0.85, 0.90]
]


def evaluate_thresholds(y_true, probabilities):
    """Compare candidate thresholds on validation data."""
    from sklearn.metrics import confusion_matrix

    results = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_true, predictions, zero_division=0
        )
        recall = recall_score(
            y_true, predictions, zero_division=0
        )
        f1 = f1_score(
            y_true, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarms": int(fp),
            "missed_failures": int(fn),
            "detected_failures": int(tp),
            "true_negatives": int(tn),
        })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(DATA_PATH)

    excluded = [
        "UDI",
        "Product ID",
        TARGET,
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(columns=excluded)
    y = df[TARGET]

    categorical_features = ["Type"]
    numerical_features = [
        column for column in X.columns
        if column not in categorical_features
    ]

    # First hold out the final 20% test partition.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Of the remaining 80%, hold out 25% for validation:
    # 60% training, 20% validation, 20% final test.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=21,
        stratify=y_train_val,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    candidates = {
        "Dummy baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Fit candidate models on training data only.
    # Select the winner using validation average precision.
    fitted_models = {}
    comparison = []

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ])

        pipeline.fit(X_train, y_train)

        val_probabilities = pipeline.predict_proba(X_val)[:, 1]
        val_predictions = (val_probabilities >= 0.50).astype(int)

        ap = average_precision_score(y_val, val_probabilities)
        roc_auc = roc_auc_score(y_val, val_probabilities)

        comparison.append({
            "model": name,
            "validation_average_precision": ap,
            "validation_roc_auc": roc_auc,
            "validation_precision_at_0.50": precision_score(
                y_val, val_predictions, zero_division=0
            ),
            "validation_recall_at_0.50": recall_score(
                y_val, val_predictions, zero_division=0
            ),
            "validation_f1_at_0.50": f1_score(
                y_val, val_predictions, zero_division=0
            ),
        })

        fitted_models[name] = pipeline

        print(
            f"{name}: validation AP={ap:.4f}, "
            f"ROC AUC={roc_auc:.4f}"
        )

    comparison_df = pd.DataFrame(comparison).sort_values(
        "validation_average_precision",
        ascending=False,
    )
    comparison_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )

    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    # Choose the threshold using validation data only.
    val_probabilities = best_model.predict_proba(X_val)[:, 1]
    threshold_df = evaluate_thresholds(y_val, val_probabilities)

    threshold_df.to_csv(
        REPORTS_DIR / "threshold_comparison.csv",
        index=False,
    )

    # F1 is the declared selection objective.
    # If multiple thresholds tie, choose the higher threshold.
    best_threshold_row = (
        threshold_df.sort_values(
            ["f1", "threshold"],
            ascending=[False, False],
        ).iloc[0]
    )
    best_threshold = float(best_threshold_row["threshold"])

    # Evaluate once on the untouched final test set.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    ).ravel()

    test_metrics = {
        "selected_model": best_name,
        "threshold": best_threshold,
        "test_records": int(len(X_test)),
        "test_failure_count": int(y_test.sum()),
        "precision": float(precision_score(
            y_test, test_predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, test_predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, test_predictions, zero_division=0
        )),
        "average_precision": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_probabilities)
        ),
        "true_negatives": int(tn),
        "false_alarms": int(fp),
        "missed_failures": int(fn),
        "detected_failures": int(tp),
    }

    # Save the exact fitted model evaluated above.
    model_path = MODELS_DIR / "predictx_model.joblib"
    joblib.dump(best_model, model_path)

    # API reads this metadata in the next integration step.
    metadata = {
        "model_name": best_name,
        "threshold": best_threshold,
        "selection_metric": "validation_average_precision",
        "threshold_selection_metric": "validation_f1",
        "random_state": RANDOM_STATE,
        "split": {
            "train_records": int(len(X_train)),
            "validation_records": int(len(X_val)),
            "test_records": int(len(X_test)),
        },
        "test_metrics": test_metrics,
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (REPORTS_DIR / "final_test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(test_metrics, file, indent=2)

    print("\n--- DATA SPLIT ---")
    print(f"Training records:   {len(X_train)}")
    print(f"Validation records: {len(X_val)}")
    print(f"Final test records: {len(X_test)}")

    print("\n--- SELECTED MODEL ---")
    print(f"Model: {best_name}")
    print(f"Validation threshold: {best_threshold:.2f}")
    print(
        "Validation F1: "
        f"{float(best_threshold_row['f1']):.4f}"
    )

    print("\n--- FINAL TEST RESULTS ---")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        target_names=["Normal", "Failure"],
        zero_division=0,
    ))

    print("\nSaved model:", model_path)
    print("Saved metadata:", metadata_path)
    print("Saved model comparison:",
          REPORTS_DIR / "model_comparison.csv")
    print("Saved threshold comparison:",
          REPORTS_DIR / "threshold_comparison.csv")
    print("Saved test metrics:",
          REPORTS_DIR / "final_test_metrics.json")


if __name__ == "__main__":
    main()
