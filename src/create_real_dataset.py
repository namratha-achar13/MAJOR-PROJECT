import numpy as np
import pandas as pd
from pathlib import Path
from feature_extraction import extract_features

SAMPLE_RATE = 20e6
WINDOW_SIZE = 400_000  # 20 ms at 20 MHz

SIGNAL_TYPES = ["wifi", "bluetooth"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"
OUTPUT_FILE = PROJECT_ROOT / "data" / "real_rf_features.csv"


def create_real_dataset():
    rows = []

    print()
    print("REAL RF DATASET CREATION")
    print("========================")
    print(f"Sample rate : {SAMPLE_RATE / 1e6:.1f} MHz")
    print(f"Window size : {WINDOW_SIZE} samples")
    print(f"Window time : {WINDOW_SIZE / SAMPLE_RATE * 1000:.1f} ms")
    print()

    for signal_type in SIGNAL_TYPES:
        signal_dir = REAL_DATA_DIR / signal_type
        capture_files = sorted(signal_dir.glob("capture_*.npy"))

        print(f"{signal_type.upper():10} captures: {len(capture_files)}")

        for capture_file in capture_files:
            signal = np.load(capture_file)

            number_of_windows = len(signal) // WINDOW_SIZE

            print(
                f"  {capture_file.name}: "
                f"{len(signal)} samples -> "
                f"{number_of_windows} windows"
            )

            for i in range(number_of_windows):
                start = i * WINDOW_SIZE
                end = start + WINDOW_SIZE

                window = signal[start:end]

                features = extract_features(
                    window,
                    sample_rate=SAMPLE_RATE
                )

                features["label"] = signal_type
                features["capture"] = capture_file.name
                features["window"] = i + 1

                rows.append(features)

    if not rows:
        raise RuntimeError("No real RF captures found.")

    dataset = pd.DataFrame(rows)

    dataset.to_csv(OUTPUT_FILE, index=False)

    print()
    print("========================")
    print("REAL RF DATASET CREATED")
    print("========================")
    print(f"Total feature windows: {len(dataset)}")

    print()
    print("Windows per class:")
    print(dataset["label"].value_counts())

    print()
    print("Features:")
    print(dataset.columns.tolist())

    print()
    print("Dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    create_real_dataset()