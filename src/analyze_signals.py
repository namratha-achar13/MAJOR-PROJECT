import numpy as np
import matplotlib.pyplot as plt


def analyze_signal(signal, sample_rate=2e6):
    """
    Perform FFT analysis on an I/Q signal.

    Returns:
        frequencies      : Frequency axis
        magnitude_db     : Signal magnitude in dB
        peak_frequency   : Strongest detected frequency
    """

    # Number of samples
    N = len(signal)

    # Perform FFT
    fft_result = np.fft.fft(signal)

    # Shift zero frequency to the centre
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate magnitude
    magnitude = np.abs(fft_shifted)

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(
        magnitude + 1e-12
    )

    # Create frequency axis
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / sample_rate
        )
    )

    # Find strongest frequency
    peak_index = np.argmax(magnitude_db)

    peak_frequency = frequencies[peak_index]

    return (
        frequencies,
        magnitude_db,
        peak_frequency
    )


# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    signal_types = [
        "wifi",
        "bluetooth",
        "fm",
        "am"
    ]

    print("RF SIGNAL ANALYSIS")
    print("==================")

    for signal_type in signal_types:

        # File location
        filename = (
            f"data/simulated_{signal_type}.npy"
        )

        # Load signal
        signal = np.load(filename)

        # Analyze signal
        (
            frequencies,
            spectrum,
            peak_frequency
        ) = analyze_signal(signal)

        print(
            f"{signal_type.upper():10} "
            f"| Peak Frequency: "
            f"{peak_frequency / 1e3:.2f} kHz"
        )

    print("==================")
    print("Analysis complete!")