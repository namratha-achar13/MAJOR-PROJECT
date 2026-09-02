import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def perform_fft(iq_signal, sample_rate=20e6):
    """
    Perform FFT on complex I/Q samples.

    Parameters:
        iq_signal: Complex I/Q samples
        sample_rate: Sampling rate in Hz

    Returns:
        frequencies: Frequency bins in Hz
        magnitude_db: FFT magnitude in dB
        peak_frequency: Strongest frequency offset in Hz
    """

    iq_signal = np.asarray(iq_signal)

    if len(iq_signal) == 0:
        raise ValueError("I/Q signal is empty.")

    N = len(iq_signal)

    # FFT
    fft_result = np.fft.fft(iq_signal)

    # Move zero frequency to the centre
    fft_shifted = np.fft.fftshift(fft_result)

    # Magnitude
    magnitude = np.abs(fft_shifted)

    # Convert magnitude to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # Frequency axis
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(N, d=1 / sample_rate)
    )

    # Find strongest frequency component
    peak_index = np.argmax(magnitude_db)

    peak_frequency = frequencies[peak_index]

    return frequencies, magnitude_db, peak_frequency


def process_iq(iq_signal, sample_rate=20e6):
    """
    Process I/Q samples using FFT.

    This function can later be imported by app.py.
    """

    return perform_fft(
        iq_signal,
        sample_rate
    )


def load_iq_file(file_path):
    """
    Load I/Q samples from a .npy file.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"I/Q file not found: {file_path}"
        )

    return np.load(file_path)


def plot_spectrum(
    frequencies,
    spectrum,
    title="ADALM-PLUTO RF Spectrum"
):
    """
    Create an RF spectrum plot.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        frequencies / 1e6,
        spectrum
    )

    ax.set_title(title)
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True)

    fig.tight_layout()

    return fig


if __name__ == "__main__":

    # Project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Real Pluto I/Q capture
    iq_file = (
        PROJECT_ROOT
        / "data"
        / "real_iq_samples.npy"
    )

    print("REAL PLUTO SDR FFT")
    print("==================")

    # Load real I/Q samples
    signal = load_iq_file(iq_file)

    print(f"Loaded samples : {len(signal)}")
    print(f"Data type      : {signal.dtype}")

    # Must match capture.py
    sample_rate = 20e6

    # Process FFT
    frequencies, spectrum, peak_frequency = process_iq(
        signal,
        sample_rate
    )

    print(
        f"Peak frequency offset: "
        f"{peak_frequency / 1e6:.6f} MHz"
    )

    # Create spectrum plot
    fig = plot_spectrum(
        frequencies,
        spectrum
    )

    # Save spectrum
    output_file = (
        PROJECT_ROOT
        / "data"
        / "real_iq_fft.png"
    )

    fig.savefig(
        output_file,
        dpi=150
    )

    print(f"Spectrum saved to: {output_file}")

    plt.show()