
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ai4i_clean.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"
PREDICTION_DIR = ROOT / "outputs" / "predictions"

TARGET = "Machine failure"

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = ["Type"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor():
    """Build preprocessing pipelines for numerical and categorical data."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}. "
            "Run python -m src.preprocess first."
        )

    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results = {}

    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Training failures: {int(y_train.sum())}")
    print(f"Test failures: {int(y_test.sum())}")

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        scores = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "failure_precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "failure_recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "failure_f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "average_precision": float(
                average_precision_score(y_test, scores)
            ),
        }

        trained_models[name] = pipeline
        results[name] = metrics

        print(f"\n--- {name} ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Failure"],
            zero_division=0,
        ))

    # Choose using failure-class F1, not accuracy alone.
    best_name = max(
        results,
        key=lambda name: results[name]["failure_f1"],
    )
    best_model = trained_models[best_name]

    best_predictions = best_model.predict(X_test)
    best_scores = best_model.predict_proba(X_test)[:, 1]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "model_name": best_name,
            "target": TARGET,
        },
        MODEL_DIR / "predictive_maintenance.joblib",
    )

    report = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "failure_count": int(y.sum()),
        "selected_model": best_name,
        "models": results,
        "confusion_matrix": confusion_matrix(
            y_test, best_predictions, labels=[0, 1]
        ).tolist(),
    }

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    test_output = X_test.copy()
    test_output["actual_failure"] = y_test.to_numpy()
    test_output["predicted_failure"] = best_predictions
    test_output["failure_score"] = best_scores

    test_output.to_csv(
        PREDICTION_DIR / "test_predictions.csv",
        index=False,
    )

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'predictive_maintenance.joblib'}")
    print(f"Saved report: {REPORT_DIR / 'metrics.json'}")
    print(f"Saved predictions: {PREDICTION_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
