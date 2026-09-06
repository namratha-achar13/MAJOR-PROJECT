# 📡 AI-Based RF Signal Classification and Direction Estimation

## 📌 Overview

This project presents an AI-based RF signal analysis system capable of capturing, processing, classifying, and analyzing radio-frequency signals using an **ADALM-PLUTO Software Defined Radio (SDR)**.

The system combines Software Defined Radio, digital signal processing, machine learning, and direction estimation techniques to analyze RF signals and provide an interactive visualization through a Streamlit dashboard.

The system performs:

- RF signal acquisition using ADALM-PLUTO SDR
- I/Q signal processing
- Fast Fourier Transform (FFT) analysis
- RF feature extraction
- Machine Learning based signal classification
- Random Forest classification
- Frequency spectrum visualization
- Spectrogram analysis
- Direction of Arrival (DOA) estimation
- Interactive Streamlit dashboard visualization

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
8. Provide an interactive dashboard for RF signal analysis.
9. Develop a system that can be extended to additional RF signal classes.

---

# 🏗️ System Architecture

The proposed system follows an end-to-end RF signal acquisition, processing, classification, visualization, and direction estimation pipeline.

```text
                         ┌──────────────────────┐
                         │    RF Environment    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   ADALM-PLUTO SDR   │
                         │   RF Signal Capture  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      I/Q Samples     │
                         │   Data Acquisition   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                   ┌────────────────────────────────┐
                   │       Signal Processing        │
                   ├────────────────────────────────┤
                   │ • FFT Analysis                  │
                   │ • Power Calculation             │
                   │ • Feature Extraction            │
                   │ • Spectrogram Generation        │
                   └───────────────┬────────────────┘
                                   │
                   ┌───────────────┼────────────────┐
                   │               │                │
                   ▼               ▼                ▼
             ┌──────────┐   ┌──────────────┐  ┌──────────────┐
             │   FFT    │   │ RF Features │  │ Spectrogram  │
             └────┬─────┘   └──────┬───────┘  └──────────────┘
                  │                │
                  │                ▼
                  │       ┌──────────────────┐
                  │       │ Random Forest ML │
                  │       │    Classifier    │
                  │       └────────┬─────────┘
                  │                │
                  │                ▼
                  │       ┌──────────────────┐
                  │       │ Signal Class +   │
                  │       │ Confidence Score │
                  │       └──────────────────┘
                  │
                  ▼
          ┌──────────────────┐
          │ Frequency Spectrum│
          │   Visualization   │
          └──────────────────┘


                         I/Q Samples
                              │
                              ▼
                   ┌────────────────────┐
                   │  Direction Scan    │
                   │  Angle-wise Power  │
                   │    Measurement     │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Maximum Power      │
                   │ Angle Detection    │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Estimated Signal   │
                   │ Direction (DOA)    │
                   └────────────────────┘


                    ┌────────────────────────┐
                    │   Streamlit Dashboard  │
                    ├────────────────────────┤
                    │ • Signal Classification│
                    │ • Confidence           │
                    │ • RF Features          │
                    │ • FFT Spectrum         │
                    │ • Spectrogram          │
                    │ • DOA Estimation       │
                    └────────────────────────┘
```

## 🔄 Overall Workflow

```text
RF Signal
    │
    ▼
ADALM-PLUTO SDR
    │
    ▼
I/Q Data Acquisition
    │
    ▼
Signal Processing
    │
    ├──────────► FFT Analysis ──────────► Frequency Spectrum
    │
    ├──────────► Feature Extraction ────► Random Forest ───► Signal Classification
    │
    ├──────────► Spectrogram ───────────► Time-Frequency Analysis
    │
    └──────────► DOA Analysis ──────────► Estimated Direction
                                                   │
                                                   ▼
                                         Streamlit Dashboard
```

## 🛠️ Technologies Used

### Programming Language
- **Python** – Core programming language used for RF signal processing, feature extraction, machine learning, and dashboard development.

### RF & SDR
- **ADALM-PLUTO SDR** – Used for RF signal acquisition and I/Q data capture.
- **I/Q Signal Processing** – Used to represent and process the acquired complex RF signals.

### Signal Processing
- **NumPy** – Numerical computing and complex I/Q signal processing.
- **SciPy** – Signal processing and spectral analysis.
- **FFT (Fast Fourier Transform)** – Used for frequency-domain analysis.
- **Spectrogram Analysis** – Used for time-frequency visualization of RF signals.

### Machine Learning
- **Scikit-learn** – Machine learning framework used for model development.
- **Random Forest Classifier** – Used for RF signal classification.
- **Joblib** – Used to save and load the trained machine learning model.

### Data Processing & Visualization
- **Pandas** – Used for dataset handling, feature processing, and CSV analysis.
- **Matplotlib** – Used for frequency spectrum, spectrogram, and DOA visualization.

### Dashboard
- **Streamlit** – Used to develop the interactive RF signal analysis dashboard.

### Development & Version Control
- **Git** – Used for version control and project management.
- **GitHub** – Used for repository hosting and team collaboration.

---

## 📂 Project Structure

```text
MAJOR-PROJECT/
│
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── doa_scan_results.csv
│   ├── real_rf_features.csv
│   ├── simulated_am.npy
│   ├── simulated_bluetooth.npy
│   ├── simulated_fm.npy
│   └── simulated_wifi.npy
│
├── models/
│   └── real_rf_model.pkl
│
└── src/
    ├── analyze_signals.py
    ├── capture.py
    ├── classifier.py
    ├── create_dataset.py
    ├── create_real_dataset.py
    ├── direction.py
    ├── evaluate_real_model.py
    ├── feature_extraction.py
    ├── fft.py
    ├── signal_simulator.py
    └── spectrogram.py
```

---

## 🔬 Signal Processing Pipeline

### 1. RF Signal Acquisition

The **ADALM-PLUTO SDR** is used to capture RF signals from the surrounding RF environment.

The SDR provides complex I/Q samples consisting of:

- In-phase (I) component
- Quadrature (Q) component

These samples are used as the primary input for subsequent signal-processing operations.

### 2. I/Q Signal Processing

The captured complex samples are processed using Python-based numerical and signal-processing techniques.

The signal processing stage prepares the raw I/Q data for:

- Frequency analysis
- Feature extraction
- Spectrogram generation
- Machine Learning classification
- Direction estimation

### 3. FFT Analysis

The **Fast Fourier Transform (FFT)** is applied to the captured I/Q signal to convert it from the time domain into the frequency domain.

FFT analysis is used to determine:

- Frequency components
- Dominant frequency
- Spectral characteristics
- Signal distribution across frequency

The resulting frequency spectrum is visualized in the Streamlit dashboard.

### 4. Feature Extraction

RF features are extracted from the captured signal and used as input to the Machine Learning model.

The feature set consists of:

```text
1. Peak Frequency
2. Power
3. Mean Magnitude
4. Standard Deviation
5. Bandwidth
6. Spectral Centroid
7. Spectral Spread
8. Spectral Flatness
9. Spectral Entropy
```

These features provide a numerical representation of the characteristics of the RF signal.

---

## 🤖 Machine Learning Classification

A **Random Forest Classifier** is used to classify RF signals based on the extracted signal features.

The trained model is stored in:

```text
models/real_rf_model.pkl
```

The classification process follows:

```text
Captured I/Q Signal
        ↓
Feature Extraction
        ↓
9-Dimensional Feature Vector
        ↓
Random Forest Classifier
        ↓
Predicted Signal Class
        ↓
Confidence Score
```

The current real-RF model contains:

```text
Wi-Fi
Bluetooth
```

The system architecture can be extended to additional RF signal classes by collecting suitable real RF data and retraining the model.

---

## 📊 Frequency Spectrum Analysis

The frequency spectrum provides a frequency-domain representation of the captured RF signal.

The processing sequence is:

```text
I/Q Samples
     ↓
FFT
     ↓
FFT Shift
     ↓
Magnitude Calculation
     ↓
Magnitude in dB
     ↓
Frequency Spectrum
```

The spectrum is displayed through the Streamlit dashboard for visual analysis of the received RF signal.

---

## 🌈 Spectrogram Analysis

A spectrogram provides a time-frequency representation of the RF signal.

It allows the user to observe how the frequency content of the signal changes over time.

The spectrogram can be used to analyze:

- Signal activity
- Frequency variations
- Signal duration
- Time-varying RF characteristics

---

## 🧭 Direction of Arrival (DOA) Estimation

The project implements a **power-based Direction of Arrival estimation technique**.

The receiving antenna is positioned at different angles and the received signal power is measured.

The process is:

```text
Antenna Position
       ↓
Angle-wise RF Measurement
       ↓
Received Power Calculation
       ↓
Power vs. Angle
       ↓
Maximum Power Detection
       ↓
Estimated Direction
```

The measured DOA results are stored in:

```text
data/doa_scan_results.csv
```

The angle corresponding to the maximum received power is selected as the estimated signal direction.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard for visualizing the RF analysis process.

The dashboard provides:

### 📡 RF Signal Capture
- ADALM-PLUTO connection
- Real I/Q signal capture
- Capture status

### 🤖 AI Classification
- Predicted RF signal type
- Classification confidence
- Class probabilities

### 📊 RF Analysis
- Peak frequency
- Signal power
- Extracted RF features

### 📈 Visualization
- Frequency spectrum
- RF spectrogram
- DOA scan graph

### 🧭 Direction Estimation
- Estimated signal direction
- Maximum received power
- Power-versus-angle visualization

---

## 🧪 Dataset

The project contains both simulated and real RF data.

### Simulated RF Data

The simulated datasets include:

```text
data/simulated_wifi.npy
data/simulated_bluetooth.npy
data/simulated_fm.npy
data/simulated_am.npy
```

These datasets are used for signal simulation, development, testing, and signal-processing experimentation.

### Real RF Dataset

The real RF feature dataset is stored in:

```text
data/real_rf_features.csv
```

The extracted real RF features are used for Machine Learning model training and evaluation.

---

## 🧠 Trained Model

The trained Random Forest model is stored at:

```text
models/real_rf_model.pkl
```

The model uses nine RF signal features:

```text
Peak Frequency
Power
Mean Magnitude
Standard Deviation
Bandwidth
Spectral Centroid
Spectral Spread
Spectral Flatness
Spectral Entropy
```

The model currently performs classification of the real RF classes available in the training dataset.

---

## 📈 Results

The developed system provides an end-to-end RF signal analysis pipeline with the following capabilities:

- Real RF I/Q signal acquisition
- Frequency-domain analysis
- RF feature extraction
- Machine Learning based signal classification
- Classification confidence estimation
- Class probability analysis
- Frequency spectrum visualization
- Spectrogram visualization
- Power-based direction estimation
- Interactive Streamlit dashboard

The system demonstrates the integration of SDR technology, digital signal processing, Machine Learning, and RF direction estimation within a unified software platform.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/namratha-achar13/MAJOR-PROJECT.git
```

### 2. Navigate to the Project Directory

```bash
cd MAJOR-PROJECT
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Dashboard

Start the Streamlit application using:

```bash
streamlit run app.py
```

If Streamlit is not recognized directly, use:

```powershell
python -m streamlit run app.py
```

The application will open in a web browser.

---

## 📡 Hardware Operation

For real RF signal acquisition:

1. Connect the ADALM-PLUTO SDR to the computer.
2. Connect the required RF receiving antenna.
3. Ensure the SDR is detected correctly.
4. Start the Streamlit dashboard.
5. Select the real RF capture option.
6. Capture the I/Q samples.
7. Process the captured signal.
8. Extract RF features.
9. Classify the signal using the trained Random Forest model.
10. Analyze the frequency spectrum and spectrogram.
11. Perform DOA analysis using the directional scan results.

---

## 🔧 Configuration

Important RF capture parameters include:

```text
Center Frequency
Sample Rate
RX Bandwidth
Number of Capture Samples
RX Gain
```

These parameters can be adjusted according to the SDR configuration and RF monitoring requirements.

---

## 🔮 Future Scope

The system can be further enhanced through:

- Addition of more RF signal classes.
- Collection of larger and more diverse real RF datasets.
- Improved RF classification accuracy.
- Real-time continuous RF monitoring.
- Multiple-SDR based direction estimation.
- Antenna-array based DOA algorithms.
- Advanced Machine Learning and Deep Learning models.
- CNN-based RF signal classification.
- Automatic RF signal detection and alerts.
- Support for a wider range of RF frequency bands.
- Improved robustness under different noise conditions.

---

## 👥 Project Team

### Major Project – AI-Based RF Signal Classification and Direction Estimation

- **Namratha Achar**
- **Praneeth N**
- **Jathin M**
- **Harrsha R**

---

## 📜 License

This project is developed for academic and educational purposes.

See the `LICENSE` file for additional information.
