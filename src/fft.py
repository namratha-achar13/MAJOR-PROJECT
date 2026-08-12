import numpy as np
import matplotlib.pyplot as plt


def perform_fft(iq_signal, sample_rate=2e6):

    N = len(iq_signal)

    # FFT
    fft_result = np.fft.fft(iq_signal)

    # Move zero frequency to centre
    fft_shifted = np.fft.fftshift(fft_result)

    # Magnitude
    magnitude = np.abs(fft_shifted)

    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-12)

    # Frequency axis
    frequencies = np.fft.fftshift(
        np.fft.fftfreq(N, d=1 / sample_rate)
    )

    # Find strongest frequency
    peak_index = np.argmax(magnitude_db)
    peak_frequency = frequencies[peak_index]

    return frequencies, magnitude_db, peak_frequency


# Load simulated signal
signal = np.load("data/simulated_wifi.npy")

# Run FFT
frequencies, spectrum, peak_frequency = perform_fft(signal)

# Print detected frequency
print("Detected Peak Frequency:", peak_frequency / 1e3, "kHz")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(frequencies / 1e6, spectrum)

plt.title("Simulated RF Signal Spectrum")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)

plt.show()