import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "real_rf_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "real_rf_model.pkl"

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

TEST_CAPTURE = "capture_005.npy"


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Real RF dataset not found: {DATA_FILE}"
        )

    dataset = pd.read_csv(DATA_FILE)

    required_columns = FEATURE_COLUMNS + [
        TARGET_COLUMN,
        "capture"
    ]

    for column in required_columns:
        if column not in dataset.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    return dataset


def train_real_model():

    dataset = load_dataset()

    print()
    print("REAL RF SIGNAL CLASSIFIER")
    print("=========================")
    print()

    print(f"Dataset samples : {len(dataset)}")
    print(
        f"Number of features: {len(FEATURE_COLUMNS)}"
    )
    print(
        f"Signal classes  : "
        f"{sorted(dataset[TARGET_COLUMN].unique())}"
    )
    print()

    # --------------------------------------------------
    # Capture-level train/test split
    # --------------------------------------------------

    train_data = dataset[
        dataset["capture"] != TEST_CAPTURE
    ].copy()

    test_data = dataset[
        dataset["capture"] == TEST_CAPTURE
    ].copy()

    print("TRAINING DATA")
    print("-------------")
    print(f"Samples: {len(train_data)}")
    print(
        train_data[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    print()

    print("TEST DATA")
    print("---------")
    print(f"Samples: {len(test_data)}")
    print(
        test_data[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    print()

    # --------------------------------------------------
    # Prepare features and labels
    # --------------------------------------------------

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    # --------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("REAL RF MODEL RESULTS")
    print("=====================")

    print(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print("Classification Report")
    print("---------------------")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[
                "bluetooth",
                "wifi"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("----------------")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[
            "bluetooth",
            "wifi"
        ]
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                "Actual Bluetooth",
                "Actual Wi-Fi"
            ],
            columns=[
                "Predicted Bluetooth",
                "Predicted Wi-Fi"
            ]
        )
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    print()
    print("FEATURE IMPORTANCE")
    print("------------------")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLUMNS
    ).sort_values(
        ascending=False
    )

    print(
        importance.round(4).to_string()
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print()
    print("Model saved to:")
    print(MODEL_FILE)

    print()
    print("=====================")
    print("Real RF training complete!")
    print("=====================")

    return model, accuracy


if __name__ == "__main__":
    train_real_model()