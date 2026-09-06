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

---

# 🏗️ System Architecture

```text
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
