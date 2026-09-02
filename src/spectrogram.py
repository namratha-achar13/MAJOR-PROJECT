import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SAMPLE_RATE = 20e6
WINDOW_SIZE = 1024
OVERLAP = 512


def generate_spectrogram(iq_signal, sample_rate=SAMPLE_RATE):
    """
    Generate a time-frequency spectrogram from complex I/Q samples.

    Parameters:
        iq_signal: Complex I/Q samples
        sample_rate: Sampling rate in Hz

    Returns:
        spectrum_db: Spectrogram magnitude in dB
        frequencies: Frequency offsets in Hz
        times: Time positions in seconds
    """

    iq_signal = np.asarray(iq_signal)

    if len(iq_signal) == 0:
        raise ValueError("I/Q signal is empty.")

    step = WINDOW_SIZE - OVERLAP

    if len(iq_signal) < WINDOW_SIZE:
        raise ValueError(
            "Not enough I/Q samples for the spectrogram."
        )

    window = np.hanning(WINDOW_SIZE)

    spectra = []
    times = []

    for start in range(
        0,
        len(iq_signal) - WINDOW_SIZE + 1,
        step
    ):
        segment = iq_signal[
            start:start + WINDOW_SIZE
        ]

        # Remove DC component
        segment = segment - np.mean(segment)

        # Apply Hann window
        windowed = segment * window

        # FFT
        fft_result = np.fft.fft(windowed)

        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_result)

        # Magnitude
        magnitude = np.abs(fft_shifted)

        # Convert to dB
        magnitude_db = 20 * np.log10(
            magnitude + 1e-12
        )

        spectra.append(magnitude_db)

        center_sample = start + WINDOW_SIZE / 2
        times.append(center_sample / sample_rate)

    spectrum_db = np.array(spectra).T

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            WINDOW_SIZE,
            d=1 / sample_rate
        )
    )

    times = np.array(times)

    return spectrum_db, frequencies, times


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


def plot_spectrogram(
    spectrum_db,
    frequencies,
    times,
    title="ADALM-PLUTO RF Spectrogram"
):
    """
    Create a time-frequency spectrogram plot.
    """

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    mesh = ax.pcolormesh(
        times * 1000,
        frequencies / 1e6,
        spectrum_db,
        shading="auto"
    )

    fig.colorbar(
        mesh,
        ax=ax,
        label="Magnitude (dB)"
    )

    ax.set_title(title)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency Offset (MHz)")

    fig.tight_layout()

    return fig


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    iq_file = (
        PROJECT_ROOT
        / "data"
        / "real_iq_samples.npy"
    )

    output_file = (
        PROJECT_ROOT
        / "data"
        / "real_iq_spectrogram.png"
    )

    print("REAL PLUTO SDR SPECTROGRAM")
    print("==========================")

    signal = load_iq_file(iq_file)

    print(f"Loaded samples : {len(signal)}")
    print(f"Data type      : {signal.dtype}")
    print(
        f"Sample rate    : "
        f"{SAMPLE_RATE / 1e6:.1f} MHz"
    )

    spectrum_db, frequencies, times = (
        generate_spectrogram(
            signal,
            SAMPLE_RATE
        )
    )

    print(
        f"Time duration  : "
        f"{len(signal) / SAMPLE_RATE * 1000:.3f} ms"
    )

    fig = plot_spectrogram(
        spectrum_db,
        frequencies,
        times
    )

    fig.savefig(
        output_file,
        dpi=150
    )

    print()
    print(
        f"Spectrogram saved to: "
        f"{output_file}"
    )

    plt.show()