import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "real_rf_features.csv"

FEATURE_COLUMNS = [
    "peak_frequency",
    "power",
    "mean_magnitude",
    "std_magnitude",
    "bandwidth",
    "spectral_centroid",
    "spectral_spread",
    "spectral_flatness",
    "spectral_entropy"
]

TARGET_COLUMN = "label"


def evaluate_model():

    dataset = pd.read_csv(DATA_FILE)

    captures = sorted(
        dataset["capture"].unique()
    )

    all_predictions = []
    all_actual = []

    print()
    print("LEAVE-ONE-CAPTURE-OUT EVALUATION")
    print("================================")
    print()

    for test_capture in captures:

        train_data = dataset[
            dataset["capture"] != test_capture
        ]

        test_data = dataset[
            dataset["capture"] == test_capture
        ]

        X_train = train_data[FEATURE_COLUMNS]
        y_train = train_data[TARGET_COLUMN]

        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data[TARGET_COLUMN]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        all_predictions.extend(
            predictions
        )

        all_actual.extend(
            y_test
        )

        print(
            f"{test_capture}: "
            f"{accuracy * 100:.2f}% "
            f"({sum(predictions == y_test)}"
            f"/{len(y_test)} correct)"
        )

    overall_accuracy = accuracy_score(
        all_actual,
        all_predictions
    )

    print()
    print("--------------------------------")
    print(
        f"Overall accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )
    print("--------------------------------")


if __name__ == "__main__":
    evaluate_model()