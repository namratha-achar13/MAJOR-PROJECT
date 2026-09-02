import numpy as np
from pathlib import Path

DEFAULT_SAMPLE_RATE = 20e6
DC_EXCLUSION_BAND = 100e3


def extract_features(iq_signal, sample_rate=DEFAULT_SAMPLE_RATE):
    iq_signal = np.asarray(iq_signal)

    if len(iq_signal) == 0:
        raise ValueError("I/Q signal is empty.")

    # --------------------------------------------------
    # Time-domain features
    # --------------------------------------------------

    power = np.mean(np.abs(iq_signal) ** 2)

    magnitude = np.abs(iq_signal)

    mean_magnitude = np.mean(magnitude)

    std_magnitude = np.std(magnitude)

    # --------------------------------------------------
    # FFT
    # --------------------------------------------------

    N = len(iq_signal)

    # Remove DC component
    iq_centered = iq_signal - np.mean(iq_signal)

    # Reduce spectral leakage
    window = np.hanning(N)

    windowed_signal = iq_centered * window

    fft_result = np.fft.fft(windowed_signal)

    fft_shifted = np.fft.fftshift(fft_result)

    spectrum = np.abs(fft_shifted) ** 2

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / sample_rate
        )
    )

    # --------------------------------------------------
    # Remove DC region
    # --------------------------------------------------

    valid_indices = (
        np.abs(frequencies) >= DC_EXCLUSION_BAND
    )

    valid_spectrum = spectrum[valid_indices]

    valid_frequencies = frequencies[valid_indices]

    if len(valid_spectrum) == 0:
        raise ValueError(
            "No frequency bins remain after DC exclusion."
        )

    # --------------------------------------------------
    # Peak frequency
    # --------------------------------------------------

    peak_index = np.argmax(valid_spectrum)

    peak_frequency = abs(
        valid_frequencies[peak_index]
    )

    # --------------------------------------------------
    # Bandwidth
    # --------------------------------------------------

    threshold = 0.25 * np.max(valid_spectrum)

    signal_indices = np.where(
        valid_spectrum >= threshold
    )[0]

    if len(signal_indices) > 1:

        bandwidth = (
            valid_frequencies[signal_indices[-1]]
            - valid_frequencies[signal_indices[0]]
        )

        bandwidth = abs(bandwidth)

    else:

        bandwidth = 0.0

    # --------------------------------------------------
    # Normalize spectral power
    # --------------------------------------------------

    total_spectral_power = np.sum(
        valid_spectrum
    )

    if total_spectral_power <= 0:

        spectral_centroid = 0.0
        spectral_spread = 0.0
        spectral_flatness = 0.0
        spectral_entropy = 0.0

    else:

        probability = (
            valid_spectrum
            / total_spectral_power
        )

        # --------------------------------------------------
        # Spectral centroid
        # --------------------------------------------------

        spectral_centroid = np.sum(
            np.abs(valid_frequencies)
            * probability
        )

        # --------------------------------------------------
        # Spectral spread
        # --------------------------------------------------

        spectral_spread = np.sqrt(
            np.sum(
                (
                    np.abs(valid_frequencies)
                    - spectral_centroid
                ) ** 2
                * probability
            )
        )

        # --------------------------------------------------
        # Spectral flatness
        # --------------------------------------------------

        positive_spectrum = (
            valid_spectrum + 1e-12
        )

        geometric_mean = np.exp(
            np.mean(
                np.log(
                    positive_spectrum
                )
            )
        )

        arithmetic_mean = np.mean(
            positive_spectrum
        )

        spectral_flatness = (
            geometric_mean
            / arithmetic_mean
        )

        # --------------------------------------------------
        # Spectral entropy
        # --------------------------------------------------

        spectral_entropy = -np.sum(
            probability
            * np.log2(
                probability + 1e-12
            )
        )

    # --------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------

    features = {

        # Existing features
        "peak_frequency": float(
            peak_frequency
        ),

        "power": float(
            power
        ),

        "mean_magnitude": float(
            mean_magnitude
        ),

        "std_magnitude": float(
            std_magnitude
        ),

        "bandwidth": float(
            bandwidth
        ),

        # New spectral features
        "spectral_centroid": float(
            spectral_centroid
        ),

        "spectral_spread": float(
            spectral_spread
        ),

        "spectral_flatness": float(
            spectral_flatness
        ),

        "spectral_entropy": float(
            spectral_entropy
        )
    }

    return features


def load_iq_file(file_path):

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"I/Q file not found: {file_path}"
        )

    return np.load(file_path)


if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__).resolve().parent.parent
    )

    iq_file = (
        PROJECT_ROOT
        / "data"
        / "real_iq_samples.npy"
    )

    print(
        "REAL PLUTO SDR FEATURE EXTRACTION"
    )

    print(
        "================================="
    )

    signal = load_iq_file(iq_file)

    print(
        f"Loaded samples : {len(signal)}"
    )

    print(
        f"Data type      : {signal.dtype}"
    )

    features = extract_features(
        signal,
        sample_rate=DEFAULT_SAMPLE_RATE
    )

    print()
    print("Extracted Features")
    print("------------------")

    for name, value in features.items():

        if "frequency" in name or name == "bandwidth":

            print(
                f"{name:20}: "
                f"{value / 1e6:.6f} MHz"
            )

        else:

            print(
                f"{name:20}: "
                f"{value:.6f}"
            )

    print()
    print("=================================")
    print(
        "Feature extraction complete!"
    )