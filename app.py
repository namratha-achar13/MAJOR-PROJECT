import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.signal_simulator import generate_signal
from src.direction import simulate_direction


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="RF Signal Analyzer",
    page_icon="📡",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("📡 RF Signal Analyzer")

st.write(
    "Software simulation of RF signal classification "
    "and direction estimation."
)

st.info(
    "Simulation Mode — RTL-SDR hardware will be "
    "integrated later."
)


# =====================================================
# SIGNAL OPTIONS
# =====================================================

signal_types = [
    "wifi",
    "bluetooth",
    "fm",
    "am"
]

selected_signal = st.selectbox(
    "Select RF Signal",
    signal_types
)


# =====================================================
# SIGNAL GENERATION
# =====================================================

if st.button("🔍 Analyze Signal"):

    # Generate signal

    signal = generate_signal(
        selected_signal
    )

    # =================================================
    # FFT
    # =================================================

    sample_rate = 2e6

    N = len(signal)

    fft_result = np.fft.fft(signal)

    fft_shifted = np.fft.fftshift(
        fft_result
    )

    spectrum = np.abs(
        fft_shifted
    )

    spectrum_db = 20 * np.log10(
        spectrum + 1e-12
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            N,
            d=1 / sample_rate
        )
    )

    # Peak frequency

    peak_index = np.argmax(
        spectrum_db
    )

    peak_frequency = (
        frequencies[peak_index]
    )


    # =================================================
    # SIGNAL FEATURES
    # =================================================

    magnitude = np.abs(signal)

    power = np.mean(
        magnitude ** 2
    )

    mean_magnitude = np.mean(
        magnitude
    )

    std_magnitude = np.std(
        magnitude
    )


    # =================================================
    # CLASSIFICATION
    # =================================================

    # Training data

    X = []
    y = []

    for signal_type in signal_types:

        training_signal = generate_signal(
            signal_type
        )

        training_magnitude = np.abs(
            training_signal
        )

        training_fft = np.fft.fft(
            training_signal
        )

        training_fft = np.fft.fftshift(
            training_fft
        )

        training_spectrum = np.abs(
            training_fft
        )

        training_frequencies = (
            np.fft.fftshift(
                np.fft.fftfreq(
                    len(training_signal),
                    d=1 / sample_rate
                )
            )
        )

        training_peak_index = np.argmax(
            training_spectrum
        )

        training_peak_frequency = abs(
            training_frequencies[
                training_peak_index
            ]
        )

        training_features = [
            training_peak_frequency,
            np.mean(
                training_magnitude ** 2
            ),
            np.mean(
                training_magnitude
            ),
            np.std(
                training_magnitude
            )
        ]

        X.append(
            training_features
        )

        y.append(
            signal_type
        )


    # Encode labels

    label_encoder = LabelEncoder()

    y_encoded = (
        label_encoder.fit_transform(y)
    )


    # Train model

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        np.array(X),
        y_encoded
    )


    # Features of current signal

    current_features = np.array([
        abs(peak_frequency),
        power,
        mean_magnitude,
        std_magnitude
    ]).reshape(1, -1)


    # Prediction

    prediction = model.predict(
        current_features
    )

    probabilities = model.predict_proba(
        current_features
    )

    predicted_signal = (
        label_encoder.inverse_transform(
            prediction
        )[0]
    )

    confidence = (
        np.max(probabilities) * 100
    )


    # =================================================
    # DOA
    # =================================================

    direction = simulate_direction()


    # =================================================
    # DISPLAY RESULTS
    # =================================================

    st.subheader("Analysis Results")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Detected Signal",
            predicted_signal.upper()
        )


    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    with col3:

        st.metric(
            "Direction",
            f"{direction:.1f}°"
        )


    st.write(
        f"**Peak Frequency:** "
        f"{abs(peak_frequency) / 1e3:.2f} kHz"
    )

    st.write(
        f"**Signal Power:** "
        f"{power:.4f}"
    )


    # =================================================
    # SPECTRUM GRAPH
    # =================================================

    st.subheader(
        "Frequency Spectrum"
    )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        frequencies / 1e6,
        spectrum_db
    )

    ax.set_xlabel(
        "Frequency (MHz)"
    )

    ax.set_ylabel(
        "Magnitude (dB)"
    )

    ax.set_title(
        "RF Signal Spectrum"
    )

    ax.grid(True)

    st.pyplot(fig)


    # =================================================
    # STATUS
    # =================================================

    st.success(
        "Signal analysis completed successfully."
    )