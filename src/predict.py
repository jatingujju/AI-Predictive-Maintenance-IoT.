
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "predictive_maintenance.joblib"


def predict_machine_failure(
    machine_type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """Predict machine failure using the trained Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run: python -m src.train"
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]

    machine_type = machine_type.upper()

    if machine_type not in ["L", "M", "H"]:
        raise ValueError("Machine type must be L, M, or H.")

    input_data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }])

    prediction = int(model.predict(input_data)[0])
    failure_probability = float(
        model.predict_proba(input_data)[0][1]
    )

    return {
        "prediction": prediction,
        "status": (
            "Failure predicted"
            if prediction == 1
            else "Normal predicted"
        ),
        "failure_probability": failure_probability,
        "model": saved["model_name"],
    }


def main():
    test_cases = [
        {
            "name": "Normal operating conditions",
            "machine_type": "M",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 100,
        },
        {
            "name": "High tool wear and torque",
            "machine_type": "L",
            "air_temperature": 303.0,
            "process_temperature": 313.0,
            "rotational_speed": 1400,
            "torque": 60.0,
            "tool_wear": 240,
        },
        {
            "name": "High rotational speed",
            "machine_type": "H",
            "air_temperature": 301.0,
            "process_temperature": 311.0,
            "rotational_speed": 2500,
            "torque": 20.0,
            "tool_wear": 50,
        },
    ]

    print("=" * 45)
    print("AI-POWERED PREDICTIVE MAINTENANCE")
    print("=" * 45)

    for case in test_cases:
        test_name = case.pop("name")
        result = predict_machine_failure(**case)

        print(f"\nTest: {test_name}")
        print(f"Predicted status: {result['status']}")
        print(
            "Failure probability: "
            f"{result['failure_probability']:.4%}"
        )
        print(f"Model: {result['model']}")

    print("\nAll test cases completed.")


if __name__ == "__main__":
    main()
