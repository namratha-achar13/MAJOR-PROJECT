import numpy as np
import adi
from pathlib import Path


# --------------------------------------------------
# Pluto SDR Configuration
# --------------------------------------------------

PLUTO_URI = "usb:1.55.5"

SAMPLE_RATE = 20_000_000           # 20 MHz
CENTER_FREQUENCY = 2_437_000_000   # 2.437 GHz
RX_BANDWIDTH = 20_000_000          # 20 MHz
DEFAULT_NUM_SAMPLES = 200_000


# Project root and data directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_OUTPUT_FILE = DATA_DIR / "real_iq_samples.npy"


# --------------------------------------------------
# Connect to Pluto
# --------------------------------------------------

def connect_pluto(uri=PLUTO_URI):
    """
    Connect to the ADALM-PLUTO SDR
    and configure the receive path.
    """

    sdr = adi.Pluto(uri=uri)

    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_lo = CENTER_FREQUENCY
    sdr.rx_rf_bandwidth = RX_BANDWIDTH

    # Automatic gain control
    sdr.gain_control_mode_chan0 = "slow_attack"

    return sdr


# --------------------------------------------------
# Signal Power
# --------------------------------------------------

def calculate_signal_power(iq_samples):
    """
    Calculate average power of complex I/Q samples.
    """

    power = np.mean(np.abs(iq_samples) ** 2)

    return float(power)


# --------------------------------------------------
# Capture I/Q
# --------------------------------------------------

def capture_iq(
    num_samples=DEFAULT_NUM_SAMPLES,
    output_file=DEFAULT_OUTPUT_FILE,
    uri=PLUTO_URI
):
    """
    Capture real complex I/Q samples from ADALM-PLUTO.

    Returns:
        numpy.ndarray: Complex I/Q samples
    """

    print("Connecting to ADALM-PLUTO...")

    sdr = connect_pluto(uri)

    print("Connected to ADALM-PLUTO")
    print()
    print("SDR Configuration")
    print("-----------------")
    print(f"Center frequency : {CENTER_FREQUENCY / 1e9:.3f} GHz")
    print(f"Sample rate      : {SAMPLE_RATE / 1e6:.1f} MHz")
    print(f"RX bandwidth     : {RX_BANDWIDTH / 1e6:.1f} MHz")
    print(f"Samples requested: {num_samples}")

    sdr.rx_buffer_size = num_samples

    print()
    print("Capturing I/Q samples...")

    iq_samples = sdr.rx()

    # Ensure consistent datatype
    iq_samples = np.asarray(iq_samples, dtype=np.complex64)

    # Calculate signal power
    power = calculate_signal_power(iq_samples)

    print()
    print("Capture successful!")
    print("-------------------")
    print(f"Number of samples : {len(iq_samples)}")
    print(f"Data type         : {iq_samples.dtype}")
    print(f"Average power     : {power:.4f}")

    print()
    print("First 10 I/Q samples:")
    print(iq_samples[:10])

    # Create data directory if necessary
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save samples
    np.save(output_path, iq_samples)

    print()
    print("Saved to:")
    print(output_path)

    return iq_samples


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    capture_iq()