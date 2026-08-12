import numpy as np


def extract_features(iq_signal, sample_rate=2e6):
    """
    Extract useful numerical features from an I/Q signal.

    Returns:
        Dictionary containing signal features.
    """

    # -----------------------------------------
    # 1. Calculate signal power
    # -----------------------------------------

    power = np.mean(np.abs(iq_signal) ** 2)

    # -----------------------------------------
    # 2. Calculate magnitude
    # -----------------------------------------

    magnitude = np.abs(iq_signal)

    mean_magnitude = np.mean(magnitude)

    std_magnitude = np.std(magnitude)

    # -----------------------------------------
    # 3. Perform FFT
    # -----------------------------------------

    N = len(iq_signal)

    fft_result = np.fft.fft(iq_signal)

    fft_shifted = np.fft.fftshift(fft_result)

    spectrum = np.abs(fft_shifted)

    # Frequency axis

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / sample_rate
        )
    )

    # -----------------------------------------
    # 4. Find peak frequency
    # -----------------------------------------

    peak_index = np.argmax(spectrum)

    peak_frequency = abs(
        frequencies[peak_index]
    )

    # -----------------------------------------
    # 5. Estimate bandwidth
    # -----------------------------------------

    threshold = 0.5 * np.max(spectrum)

    signal_indices = np.where(
        spectrum >= threshold
    )[0]

    if len(signal_indices) > 0:

        bandwidth = (
            frequencies[signal_indices[-1]]
            - frequencies[signal_indices[0]]
        )

        bandwidth = abs(bandwidth)

    else:

        bandwidth = 0

    # -----------------------------------------
    # Store features
    # -----------------------------------------

    features = {
        "peak_frequency": peak_frequency,
        "power": power,
        "mean_magnitude": mean_magnitude,
        "std_magnitude": std_magnitude,
        "bandwidth": bandwidth
    }

    return features


# =====================================================
# TEST THE FEATURE EXTRACTION
# =====================================================

if __name__ == "__main__":

    signal_types = [
        "wifi",
        "bluetooth",
        "fm",
        "am"
    ]

    print("RF SIGNAL FEATURE EXTRACTION")
    print("============================")

    for signal_type in signal_types:

        # Load signal

        filename = (
            f"data/simulated_{signal_type}.npy"
        )

        signal = np.load(filename)

        # Extract features

        features = extract_features(signal)

        print("\nSignal:", signal_type.upper())

        print(
            "Peak Frequency:",
            round(
                features["peak_frequency"] / 1e3,
                2
            ),
            "kHz"
        )

        print(
            "Power:",
            round(
                features["power"],
                4
            )
        )

        print(
            "Mean Magnitude:",
            round(
                features["mean_magnitude"],
                4
            )
        )

        print(
            "Standard Deviation:",
            round(
                features["std_magnitude"],
                4
            )
        )

        print(
            "Bandwidth:",
            round(
                features["bandwidth"] / 1e3,
                2
            ),
            "kHz"
        )

    print("\n============================")
    print("Feature extraction complete!")