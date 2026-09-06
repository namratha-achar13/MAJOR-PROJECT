# 📡 AI-Based RF Signal Classification and Direction Estimation

## 📌 Overview

This project presents an AI-based RF signal analysis system capable of capturing, processing, classifying, and analyzing radio-frequency signals using an **ADALM-PLUTO Software Defined Radio (SDR)**.

The system combines:

- Software Defined Radio (SDR)
- I/Q signal acquisition
- Fast Fourier Transform (FFT)
- RF feature extraction
- Machine Learning
- Random Forest classification
- Frequency spectrum analysis
- Spectrogram visualization
- Direction of Arrival (DOA) estimation
- Streamlit-based visualization dashboard

The objective is to develop a software-driven RF monitoring system that can identify RF signal types and estimate the direction from which a signal is received.

---

# 🎯 Objectives

The main objectives of this project are:

1. Capture real RF signals using an SDR.
2. Process raw I/Q samples obtained from the SDR.
3. Analyze signals in the frequency domain using FFT.
4. Extract meaningful RF signal features.
5. Classify RF signals using a trained Machine Learning model.
6. Visualize frequency spectra and spectrograms.
7. Estimate signal direction using received signal power.
8. Provide an interactive dashboard for RF analysis.
9. Develop a system that can be extended to additional RF signal classes.

                    RF Signal
                        │
                        ▼
              ┌─────────────────┐
              │   ADALM-PLUTO   │
              │       SDR       │
              └────────┬────────┘
                       │
                       ▼
                 I/Q Samples
                       │
                       ▼
              ┌─────────────────┐
              │ Signal Processing│
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        FFT       Feature       Spectrogram
                   Extraction
          │            │
          │            ▼
          │     Random Forest
          │       Classifier
          │            │
          │            ▼
          │     Signal Class
          │
          ▼
   Frequency Spectrum

                       │
                       ▼
                DOA Estimation
                       │
                       ▼
               Estimated Angle

                       │
                       ▼
              Streamlit Dashboard

## 🛠️ Technologies Used

### Programming Language
- **Python** – Core programming language used for signal processing, feature extraction, machine learning, and dashboard development.

### RF & SDR
- **ADALM-PLUTO SDR** – Used for real-time RF signal acquisition and I/Q data capture.
- **I/Q Signal Processing** – Used to represent and process captured RF signals.

### Signal Processing
- **NumPy** – Numerical computing and complex I/Q signal processing.
- **SciPy** – Signal processing and spectral analysis.
- **FFT (Fast Fourier Transform)** – Used for frequency-domain analysis.
- **Spectrogram Analysis** – Used for time-frequency visualization of RF signals.

### Machine Learning
- **Scikit-learn** – Machine learning framework.
- **Random Forest Classifier** – Used for RF signal classification.
- **Joblib** – Used to save and load the trained machine learning model.

### Data Processing & Visualization
- **Pandas** – Dataset handling, feature processing, and CSV analysis.
- **Matplotlib** – Frequency spectrum, spectrogram, and DOA visualization.

### Dashboard
- **Streamlit** – Used to develop the interactive RF signal analysis dashboard.

### Development & Version Control
- **Git** – Version control and project management.
- **GitHub** – Repository hosting and collaboration.
