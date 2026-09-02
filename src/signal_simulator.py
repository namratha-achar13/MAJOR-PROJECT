import numpy as np
from pathlib import Path


SAMPLE_RATE = 20e6

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_signal(
    signal_type="wifi",
    num_samples=10000,
    sample_rate=SAMPLE_RATE
):
    """
    Generate a simulated RF I/Q signal.

    Parameters:
        signal_type  : wifi, bluetooth, fm, or am
        num_samples  : Number of samples
        sample_rate  : Sampling rate in Hz

    Returns:
        Complex I/Q samples
    """

    t = np.arange(num_samples) / sample_rate

    noise = 0.05 * (
        np.random.randn(num_samples)
        + 1j * np.random.randn(num_samples)
    )

    # ------------------------------------------------
    # Wi-Fi-like signal
    # ------------------------------------------------
    if signal_type.lower() == "wifi":

        frequency = 2e6

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

        frequency = 5e6

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

        carrier_frequency = 3e6
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

        carrier_frequency = 7e6
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

    signal_types = [
        "wifi",
        "bluetooth",
        "fm",
        "am"
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("RF Signal Dataset Generation")
    print("============================")
    print(f"Sample rate: {SAMPLE_RATE / 1e6:.1f} MHz")
    print()

    for signal_type in signal_types:

        signal = generate_signal(
            signal_type=signal_type,
            sample_rate=SAMPLE_RATE
        )

        filename = (
            DATA_DIR / f"simulated_{signal_type}.npy"
        )

        np.save(filename, signal)

        print(
            f"{signal_type.upper():10} "
            f"signal saved | "
            f"Samples: {len(signal)}"
        )

    print("============================")
    print("All simulated RF signals generated!")