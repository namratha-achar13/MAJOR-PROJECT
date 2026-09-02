import numpy as np
import pandas as pd
from pathlib import Path

from feature_extraction import extract_features


# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = Path("../data")

SIGNAL_TYPES = [
    "wifi",
    "bluetooth",
    "fm",
    "am"
]

WINDOW_SIZE = 1000


# ==========================================
# CREATE DATASET
# ==========================================

rows = []

for signal_type in SIGNAL_TYPES:

    filename = DATA_DIR / f"simulated_{signal_type}.npy"

    signal = np.load(filename)

    print(
        f"{signal_type.upper():10} "
        f"Total samples: {len(signal)}"
    )

    # Split signal into windows

    number_of_windows = len(signal) // WINDOW_SIZE

    for i in range(number_of_windows):

        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE

        window = signal[start:end]

        # Extract features

        features = extract_features(window)

        # Add label

        features["label"] = signal_type

        rows.append(features)


# ==========================================
# CREATE DATAFRAME
# ==========================================

dataset = pd.DataFrame(rows)


# ==========================================
# SAVE DATASET
# ==========================================

output_file = DATA_DIR / "rf_features.csv"

dataset.to_csv(output_file, index=False)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("RF DATASET CREATED")
print("==========================================")

print("Total samples:", len(dataset))

print("\nSamples per class:")
print(dataset["label"].value_counts())

print("\nFeatures:")
print(dataset.columns.tolist())

print("\nDataset saved to:")
print(output_file)
