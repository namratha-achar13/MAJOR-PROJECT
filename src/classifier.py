import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def extract_features(iq_signal, sample_rate=2e6):

    # Signal magnitude
    magnitude = np.abs(iq_signal)

    # Signal power
    power = np.mean(magnitude ** 2)

    # Mean magnitude
    mean_magnitude = np.mean(magnitude)

    # Standard deviation
    std_magnitude = np.std(magnitude)

    # FFT
    N = len(iq_signal)

    fft_result = np.fft.fft(iq_signal)

    fft_shifted = np.fft.fftshift(fft_result)

    spectrum = np.abs(fft_shifted)

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / sample_rate
        )
    )

    # Peak frequency
    peak_index = np.argmax(spectrum)

    peak_frequency = abs(
        frequencies[peak_index]
    )

    return [
        peak_frequency,
        power,
        mean_magnitude,
        std_magnitude
    ]


# =====================================================
# CREATE TRAINING DATA
# =====================================================

signal_types = [
    "wifi",
    "bluetooth",
    "fm",
    "am"
]

X = []
y = []


for signal_type in signal_types:

    filename = (
        f"data/simulated_{signal_type}.npy"
    )

    signal = np.load(filename)

    features = extract_features(signal)

    X.append(features)
    y.append(signal_type)


# Convert to arrays

X = np.array(X)
y = np.array(y)


# =====================================================
# TRAIN CLASSIFIER
# =====================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X, y_encoded)


# =====================================================
# TEST THE MODEL
# =====================================================

print("RF SIGNAL CLASSIFIER")
print("====================")

for signal_type in signal_types:

    filename = (
        f"data/simulated_{signal_type}.npy"
    )

    signal = np.load(filename)

    features = extract_features(signal)

    features_array = np.array(
        features
    ).reshape(1, -1)

    prediction = model.predict(
        features_array
    )

    probabilities = model.predict_proba(
        features_array
    )

    predicted_label = (
        label_encoder.inverse_transform(
            prediction
        )[0]
    )

    confidence = (
        np.max(probabilities) * 100
    )

    print(
        f"{signal_type.upper():10} → "
        f"{predicted_label.upper():10} "
        f"Confidence: {confidence:.2f}%"
    )


print("====================")
print("Classification complete!")