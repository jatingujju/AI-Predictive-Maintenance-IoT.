
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ai4i+2020+predictive+maintenance+dataset"
    / "ai4i2020.csv"
)

PROCESSED_PATH = ROOT / "data" / "processed" / "ai4i_clean.csv"

TARGET = "Machine failure"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate AI4I records and prepare modeling features."""

    required = FEATURES + [TARGET]
    missing_columns = set(required) - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = data[required].copy()

    # Validate the target label.
    result = result.dropna(subset=required)
    result = result[result[TARGET].isin([0, 1])].copy()
    result[TARGET] = result[TARGET].astype(int)

    # Validate the categorical machine type.
    valid_types = ["L", "M", "H"]
    result = result[result["Type"].isin(valid_types)].copy()

    # Ensure sensor readings are numeric.
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for column in numeric_features:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    result = result.dropna(subset=numeric_features)

    if result.empty:
        raise ValueError("No valid records remain after preprocessing.")

    # Deliberately exclude identifiers and failure-mode columns:
    # UDI, Product ID, TWF, HDF, PWF, OSF, RNF.
    # This prevents identifiers and direct failure-mode labels
    # from being used as model inputs.

    return result.reset_index(drop=True)


def main():
    data = pd.read_csv(DATA_PATH)
    cleaned = preprocess_data(data)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessing completed successfully.")
    print(f"Original rows: {len(data)}")
    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Missing values after cleaning: {cleaned.isnull().sum().sum()}")
    print(f"Saved file: {PROCESSED_PATH}")

    print("\nFailure distribution:")
    print(cleaned[TARGET].value_counts().sort_index())

    print("\nModel input features:")
    print(FEATURES)


if __name__ == "__main__":
    main()
