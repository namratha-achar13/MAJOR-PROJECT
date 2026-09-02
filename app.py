import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from src.capture import connect_pluto
from src.fft import process_iq
from src.feature_extraction import extract_features
from src.spectrogram import generate_spectrogram


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "real_rf_model.pkl"
DOA_FILE = PROJECT_ROOT / "data" / "doa_scan_results.csv"


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 20_000_000
CENTER_FREQUENCY = 2_437_000_000
RX_BANDWIDTH = 20_000_000
CAPTURE_SAMPLES = 200_000

FEATURE_COLUMNS = [
    "peak_frequency",
    "power",
    "mean_magnitude",
    "std_magnitude",
    "bandwidth",
    "spectral_centroid",
    "spectral_spread",
    "spectral_flatness",
    "spectral_entropy",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Based RF Signal Analyzer",
    page_icon="📡",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📡 AI-Based RF Signal Classification & Direction Estimation")

st.write(
    "Real RF signal analysis using ADALM-PLUTO SDR, "
    "FFT, feature extraction, machine learning, "
    "and power-based direction estimation."
)

st.info(
    "Hardware Mode: ADALM-PLUTO tuned to 2.437 GHz "
    "(Wi-Fi Channel 6)."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("SDR Configuration")

st.sidebar.write(f"Center Frequency: {CENTER_FREQUENCY / 1e9:.3f} GHz")
st.sidebar.write(f"Sample Rate: {SAMPLE_RATE / 1e6:.1f} MHz")
st.sidebar.write(f"RX Bandwidth: {RX_BANDWIDTH / 1e6:.1f} MHz")
st.sidebar.write(f"Capture Samples: {CAPTURE_SAMPLES:,}")

st.sidebar.markdown("---")

st.sidebar.write("### Pipeline")

st.sidebar.write("RF Signal")
st.sidebar.write("↓")
st.sidebar.write("ADALM-PLUTO")
st.sidebar.write("↓")
st.sidebar.write("I/Q Samples")
st.sidebar.write("↓")
st.sidebar.write("FFT")
st.sidebar.write("↓")
st.sidebar.write("Feature Extraction")
st.sidebar.write("↓")
st.sidebar.write("Random Forest")
st.sidebar.write("↓")
st.sidebar.write("Signal Classification")
st.sidebar.write("↓")
st.sidebar.write("Power-Based DOA")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return joblib.load(MODEL_FILE)


# ============================================================
# SESSION STATE
# ============================================================

if "iq_signal" not in st.session_state:
    st.session_state.iq_signal = None

if "features" not in st.session_state:
    st.session_state.features = None

if "spectrum" not in st.session_state:
    st.session_state.spectrum = None

if "frequencies" not in st.session_state:
    st.session_state.frequencies = None

if "peak_frequency" not in st.session_state:
    st.session_state.peak_frequency = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None


# ============================================================
# MODEL STATUS
# ============================================================

try:
    model = load_model()
    model_status = True
except Exception as error:
    model = None
    model_status = False
    st.error(f"Model loading error: {error}")


# ============================================================
# CAPTURE SECTION
# ============================================================

st.header("1. Real RF Signal Capture")

st.write(
    "Capture real I/Q samples from the connected ADALM-PLUTO."
)

if st.button("📡 Capture Real RF Signal", use_container_width=True):

    if model is None:
        st.error("Machine-learning model could not be loaded.")
    else:

        try:
            with st.spinner("Connecting to ADALM-PLUTO and capturing I/Q data..."):

                sdr = connect_pluto()

                sdr.rx_buffer_size = CAPTURE_SAMPLES

                iq_signal = np.asarray(
                    sdr.rx(),
                    dtype=np.complex64
                )

                try:
                    sdr.rx_destroy_buffer()
                except Exception:
                    pass

            st.session_state.iq_signal = iq_signal

            st.success(
                f"Capture successful: {len(iq_signal):,} I/Q samples"
            )

        except Exception as error:

            st.error(
                f"SDR capture failed: {error}"
            )


# ============================================================
# PROCESS CAPTURE
# ============================================================

if st.session_state.iq_signal is not None:

    signal = st.session_state.iq_signal

    st.header("2. RF Signal Processing")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Samples",
            f"{len(signal):,}"
        )

    with col2:
        st.metric(
            "Duration",
            f"{len(signal) / SAMPLE_RATE * 1000:.2f} ms"
        )

    with col3:
        signal_power = float(
            np.mean(np.abs(signal) ** 2)
        )

        st.metric(
            "Average Power",
            f"{signal_power:.2f}"
        )


    # ========================================================
    # FFT
    # ========================================================

    frequencies, spectrum, peak_frequency = process_iq(
        signal,
        SAMPLE_RATE
    )

    st.session_state.frequencies = frequencies
    st.session_state.spectrum = spectrum
    st.session_state.peak_frequency = peak_frequency


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    features = extract_features(
        signal,
        sample_rate=SAMPLE_RATE
    )

    st.session_state.features = features


    # ========================================================
    # MACHINE LEARNING CLASSIFICATION
    # ========================================================

    st.header("3. AI Signal Classification")

    feature_vector = pd.DataFrame(
        [[features[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    try:

        prediction = model.predict(feature_vector)[0]

        probabilities = model.predict_proba(
            feature_vector
        )[0]

        classes = model.classes_

        probability_dict = dict(
            zip(classes, probabilities)
        )

        confidence = float(
            np.max(probabilities) * 100
        )

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Detected Signal",
                prediction.upper()
            )

        with col2:

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

        st.caption(
            "Classification uses the trained real-RF Random Forest model."
        )

        # Probability display

        st.subheader("Class Probabilities")

        probability_table = pd.DataFrame(
            {
                "Signal": classes,
                "Probability (%)": [
                    probability_dict[c] * 100
                    for c in classes
                ],
            }
        )

        st.dataframe(
            probability_table,
            hide_index=True,
            use_container_width=True
        )

    except Exception as error:

        st.error(
            f"Classification failed: {error}"
        )


    # ========================================================
    # EXTRACTED FEATURES
    # ========================================================

    st.subheader("Extracted RF Features")

    feature_display = pd.DataFrame(
        {
            "Feature": list(features.keys()),
            "Value": list(features.values()),
        }
    )

    st.dataframe(
        feature_display,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # FFT SPECTRUM
    # ========================================================

    st.header("4. Frequency Spectrum")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        frequencies / 1e6,
        spectrum
    )

    ax.set_xlabel(
        "Frequency Offset (MHz)"
    )

    ax.set_ylabel(
        "Magnitude (dB)"
    )

    ax.set_title(
        "ADALM-PLUTO RF Spectrum"
    )

    ax.grid(True)

    st.pyplot(fig)

    plt.close(fig)


    # ========================================================
    # SPECTROGRAM
    # ========================================================

    st.header("5. RF Spectrogram")

    try:

        spectrum_db, spec_frequencies, times = generate_spectrogram(
            signal,
            SAMPLE_RATE
        )

        fig2, ax2 = plt.subplots(
            figsize=(12, 5)
        )

        mesh = ax2.pcolormesh(
            times * 1000,
            spec_frequencies / 1e6,
            spectrum_db,
            shading="auto"
        )

        fig2.colorbar(
            mesh,
            ax=ax2,
            label="Magnitude (dB)"
        )

        ax2.set_xlabel(
            "Time (ms)"
        )

        ax2.set_ylabel(
            "Frequency Offset (MHz)"
        )

        ax2.set_title(
            "RF Signal Spectrogram"
        )

        st.pyplot(fig2)

        plt.close(fig2)

    except Exception as error:

        st.warning(
            f"Spectrogram could not be generated: {error}"
        )


# ============================================================
# DOA SECTION
# ============================================================

st.header("6. Direction Estimation")

st.write(
    "Power-based direction estimation using a directional RX antenna."
)

if DOA_FILE.exists():

    try:

        doa_data = pd.read_csv(
            DOA_FILE
        )

        if not doa_data.empty:

            best_row = doa_data.loc[
                doa_data["power"].idxmax()
            ]

            estimated_direction = float(
                best_row["angle"]
            )

            maximum_power = float(
                best_row["power"]
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Estimated Direction",
                    f"{estimated_direction:+.0f}°"
                )

            with col2:

                st.metric(
                    "Maximum Median Power",
                    f"{maximum_power:.2f}"
                )

            st.subheader(
                "Direction Scan"
            )

            fig3, ax3 = plt.subplots(
                figsize=(10, 5)
            )

            ax3.plot(
                doa_data["angle"],
                doa_data["power"],
                marker="o"
            )

            ax3.axvline(
                estimated_direction,
                linestyle="--",
                label=f"Estimated: {estimated_direction:+.0f}°"
            )

            ax3.set_xlabel(
                "Antenna Angle (degrees)"
            )

            ax3.set_ylabel(
                "Median Received Power"
            )

            ax3.set_title(
                "Power-Based Direction Estimation"
            )

            ax3.grid(True)

            ax3.legend()

            st.pyplot(fig3)

            plt.close(fig3)

            st.caption(
                "Direction is estimated from the angle having the "
                "maximum measured median received power."
            )

        else:

            st.warning(
                "DOA result file is empty."
            )

    except Exception as error:

        st.error(
            f"Could not read DOA results: {error}"
        )

else:

    st.info(
        "No DOA scan result found. Run src/direction.py "
        "to perform a real directional scan."
    )


# ============================================================
# FINAL STATUS
# ============================================================

if st.session_state.iq_signal is not None:

    st.success(
        "Real RF signal analysis pipeline completed successfully."
    )

else:

    st.info(
        "Press 'Capture Real RF Signal' to begin."
    )