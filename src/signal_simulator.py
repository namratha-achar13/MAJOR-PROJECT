import numpy as np


def generate_signal(signal_type="wifi", num_samples=10000, sample_rate=2e6):
    """
    Generate a simulated RF I/Q signal.

    Parameters:
        signal_type  : wifi, bluetooth, fm, or am
        num_samples  : Number of samples
        sample_rate  : Sampling rate in Hz

    Returns:
        Complex I/Q samples
    """

    # Time axis
    t = np.arange(num_samples) / sample_rate

    # Random noise
    noise = 0.05 * (
        np.random.randn(num_samples)
        + 1j * np.random.randn(num_samples)
    )

    # ------------------------------------------------
    # Wi-Fi-like signal
    # ------------------------------------------------
    if signal_type.lower() == "wifi":

        frequency = 200e3

        carrier = np.exp(
            2j * np.pi * frequency * t
        )

        modulation = 1 + 0.5 * np.sin(
            2 * np.pi * 2e3 * t
        )

        iq_signal = modulation * carrier + noise

    # ------------------------------------------------
    # Bluetooth-like signal
    # ------------------------------------------------
    elif signal_type.lower() == "bluetooth":

        frequency = 500e3

        modulation = 10e3 * np.sin(
            2 * np.pi * 1e3 * t
        )

        phase = (
            2 * np.pi * frequency * t
            + modulation
        )

        iq_signal = np.exp(1j * phase) + noise

    # ------------------------------------------------
    # FM-like signal
    # ------------------------------------------------
    elif signal_type.lower() == "fm":

        carrier_frequency = 300e3
        modulation_frequency = 5e3

        phase = (
            2 * np.pi * carrier_frequency * t
            + 2 * np.sin(
                2 * np.pi * modulation_frequency * t
            )
        )

        iq_signal = np.exp(1j * phase) + noise

    # ------------------------------------------------
    # AM-like signal
    # ------------------------------------------------
    elif signal_type.lower() == "am":

        carrier_frequency = 700e3
        modulation_frequency = 5e3

        carrier = np.exp(
            2j * np.pi * carrier_frequency * t
        )

        modulation = 1 + 0.7 * np.sin(
            2 * np.pi * modulation_frequency * t
        )

        iq_signal = modulation * carrier + noise

    else:

        raise ValueError(
            "Unknown signal type. "
            "Use: wifi, bluetooth, fm, or am."
        )

    return iq_signal


# ====================================================
# MAIN PROGRAM
# ====================================================

if __name__ == "__main__":

    # Signal types we want to simulate
    signal_types = [
        "wifi",
        "bluetooth",
        "fm",
        "am"
    ]

    print("RF Signal Dataset Generation")
    print("============================")

    # Generate each signal
    for signal_type in signal_types:

        signal = generate_signal(
            signal_type=signal_type
        )

        # File name
        filename = (
            f"data/simulated_{signal_type}.npy"
        )

        # Save I/Q samples
        np.save(filename, signal)

        print(
            f"{signal_type.upper():10} "
            f"signal saved | "
            f"Samples: {len(signal)}"
        )

    print("============================")
    print("All simulated RF signals generated!")