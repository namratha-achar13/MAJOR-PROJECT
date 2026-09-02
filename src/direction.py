import adi
import numpy as np
from pathlib import Path

PLUTO_URI = "usb:1.55.5"
CENTER_FREQUENCY = 2_437_000_000
SAMPLE_RATE = 30_720_000
RX_BANDWIDTH = 18_000_000
SAMPLES_PER_READING = 100_000
READINGS_PER_ANGLE = 10

ANGLES = [-60, -30, 0, 30, 60, 90]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = PROJECT_ROOT / "data" / "doa_scan_results.csv"


def connect_pluto():
    sdr = adi.Pluto(uri=PLUTO_URI)

    sdr.rx_lo = CENTER_FREQUENCY
    sdr.rx_buffer_size = SAMPLES_PER_READING

    return sdr


def measure_power(sdr):
    samples = np.asarray(sdr.rx(), dtype=np.complex64)
    power = np.mean(np.abs(samples) ** 2)
    return float(power)


def measure_angle(sdr, angle):
    readings = []

    print(f"\nMeasuring angle {angle:+d} degrees...")

    for i in range(READINGS_PER_ANGLE):
        power = measure_power(sdr)
        readings.append(power)
        print(f"  Reading {i + 1:02d}: {power:.2f}")

    median_power = float(np.median(readings))

    print(f"  Median power: {median_power:.2f}")

    return median_power


def run_doa_scan():
    print()
    print("REAL RF POWER-BASED DOA SCAN")
    print("============================")
    print(f"Center frequency : {CENTER_FREQUENCY / 1e9:.3f} GHz")
    print(f"Sample rate      : {SAMPLE_RATE / 1e6:.2f} MHz")
    print(f"Readings/angle   : {READINGS_PER_ANGLE}")
    print()
    print("IMPORTANT:")
    print("Rotate the directional antenna to each requested angle.")
    print("Keep the phone fixed.")
    print()

    sdr = connect_pluto()

    results = []

    for angle in ANGLES:
        input(
            f"Set antenna to {angle:+d} degrees and press ENTER..."
        )

        median_power = measure_angle(sdr, angle)

        results.append((angle, median_power))

    sdr.rx_destroy_buffer()

    print()
    print("============================")
    print("DOA SCAN RESULTS")
    print("============================")

    for angle, power in results:
        print(f"{angle:+4d} degrees : {power:.2f}")

    best_angle, best_power = max(results, key=lambda x: x[1])

    print()
    print(f"Estimated direction: {best_angle:+d} degrees")
    print(f"Maximum median power: {best_power:.2f}")

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULT_FILE, "w") as file:
        file.write("angle,power\n")

        for angle, power in results:
            file.write(f"{angle},{power}\n")

    print()
    print(f"Results saved to: {RESULT_FILE}")
    print()
    print("DOA SCAN COMPLETE")


if __name__ == "__main__":
    run_doa_scan()