
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "reports" / "metrics.json"
PREDICTIONS_PATH = ROOT / "outputs" / "predictions" / "test_predictions.csv"
IMAGE_DIR = ROOT / "images"

def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    # Chart 1: Compare model F1-scores
    model_names = list(report["models"].keys())
    f1_scores = [
        report["models"][name]["failure_f1"]
        for name in model_names
    ]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=model_names, y=f1_scores)
    plt.title("Model Comparison: Failure F1-Score")
    plt.ylabel("Failure F1-Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Chart 2: Actual versus predicted failures
    actual = predictions["actual_failure"]
    predicted = predictions["predicted_failure"]
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Failure"]
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Chart 3: Actual machine failure distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(
        x=predictions["actual_failure"],
        hue=predictions["actual_failure"],
        legend=False
    )
    plt.xticks([0, 1], ["Normal", "Failure"])
    plt.title("Actual Machine Failure Distribution")
    plt.xlabel("Machine Status")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "failure_distribution.png", dpi=150)
    plt.close()

    print("Evaluation visualizations created successfully:")
    print(IMAGE_DIR / "model_comparison.png")
    print(IMAGE_DIR / "confusion_matrix.png")
    print(IMAGE_DIR / "failure_distribution.png")

if __name__ == "__main__":
    main()
