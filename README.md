# 🩺 BioSignal Multi-Viewer Platform

A Flask-based web application that brings together **four signal processing & visualization modules** in one unified interface. Built for biomedical engineering coursework (SBEG205 — Spring 2026, Team 15).

---

## 📋 Table of Contents

- [Modules Overview](#-modules-overview)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Project Structure](#-project-structure)
- [Modules in Detail](#-modules-in-detail)

---

## 🧩 Modules Overview

| Module | Route | Description |
|--------|-------|-------------|
| 📊 **Medical Signal Viewer** | `/` | ECG/EEG signal viewer with AI & ML classification |
| 📈 **Stock Market Dashboard** | `/stock-dashboard` | Real-time stock quotes, watchlist & price charts |
| 🔊 **Acoustic Signal Lab** | `/acoustic-lab` | Doppler simulator, vehicle speed estimator & drone detector |
| 🧬 **Microbiome Signals** | `/microbiome` | Gut microbiome abundance profiling & patient risk estimation |

---

## 🛠 Tech Stack

**Backend:** Python 3, Flask, NumPy, Pandas, PyEDFLib, WFDB, yfinance

**Frontend:** HTML5, CSS3, JavaScript, Chart.js, Plotly.js, PapaParse

**AI/ML Models:** PyTorch (ECGNet via ecglib, EEGNet via Braindecode), scikit-learn–style classic ML detectors

---

## ⚙ Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd task01-signal-viewer-sbeg205_spring26_team15-main
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install flask numpy pandas pyedflib wfdb yfinance torch ecglib braindecode
   ```

---

## 🚀 Running the App

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

Navigate between modules using the buttons on the home page or by going directly to the routes listed above.

---

## 📁 Project Structure

```
├── app.py                     # Flask server — all routes & API endpoints
├── requirements.txt           # Python dependencies
├── ECG.csv                    # Sample ECG data
├── generate_hmp_data.py       # One-time script to generate microbiome CSV
│
├── models/                    # AI & Classic ML classifiers
│   ├── ecg_classifier.py      # ECGNet (ResNet1D50) — 4 pathology models
│   ├── eeg_classifier.py      # EEGNet (Braindecode) classifier
│   ├── ml_detector.py         # Classic ML feature-based detection
│   ├── ecgnet_weights.pt      # Pretrained ECG model weights
│   └── eegnet_weights.pt      # Pretrained EEG model weights
│
├── templates/                 # HTML pages
│   ├── index.html             # Medical Signal Viewer (main page)
│   ├── stock-dashboard.html   # Stock Market Dashboard
│   ├── acoustic-lab.html      # Acoustic Signal Processing Lab
│   └── microbiome.html        # Microbiome Signals
│
├── static/
│   ├── css/                   # Stylesheets
│   ├── data/                  # Static data files (microbiome CSV)
│   └── js/                    # JavaScript modules
│       ├── script.js              # Stock dashboard logic
│       ├── fft.js                 # FFT implementation
│       ├── doppler-simulator.js   # Doppler effect audio synthesis
│       ├── doppler-analyzer.js    # Vehicle speed estimation from audio
│       ├── drone-detector.js      # Drone sound detection via spectral analysis
│       ├── dataLoader.js          # CSV parser for microbiome data
│       ├── charts.js              # Microbiome chart renderers (heatmap, bar, pie, diversity)
│       ├── patientProfiler.js     # Patient risk profiler from microbiome signature
│       └── microbiome-app.js      # Microbiome page controller
│
└── utils/                     # Utility modules
```

---

## 🔍 Modules in Detail

### 📊 1. Medical Signal Viewer — `/`

The core module. Upload ECG or EEG signals and explore them interactively.

**Supported file formats:** `.csv`, `.edf`, `.dat` (PhysioNet WFDB)

**Features:**
- **Multi-window support** — open multiple signal viewers side-by-side
- **Plot types:** Signal vs Time, XOR Graph, Channel vs Channel (Lissajous), Polar Plot, Polar Ratio, Recurrence Plot
- **View modes:** Combined (overlaid) or Split (one chart per channel)
- **Interactive controls:** Play/pause animation, zoom (scroll), pan (drag), double-click to reset, speed & window size sliders
- **Per-channel customization:** Color picker, line width, visibility toggle
- **AI Classification:** Deep learning models (ECGNet ResNet1D50 for ECG, EEGNet for EEG) predict pathologies with probability bars
- **Classic ML Classification:** Feature-based detectors (HRV analysis, spectral features, statistical metrics) for comparison
- **Auto-detection:** Automatically detects ECG vs EEG based on channel names

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload signal files |
| `GET` | `/api/all_data` | Retrieve loaded signal data |
| `GET` | `/api/classify` | Run AI + ML classification |
| `GET` | `/api/set_signal_type` | Override auto-detected signal type |

---

### 📈 2. Stock Market Dashboard — `/stock-dashboard`

Real-time stock market data powered by **yfinance** (Yahoo Finance).

**Features:**
- **Search & Watchlist** — search for any stock ticker, add to your watchlist
- **Live Quotes** — current price, change, change %, with color-coded indicators
- **Bulk Quotes** — fetch up to 50 symbols in a single request
- **Price Charts** — interactive historical charts with timeframes: 1W, 1M, 3M, 1Y

**API Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/stocks/quote?symbol=AAPL` | Single stock quote |
| `GET` | `/api/stocks/bulk?symbols=AAPL,MSFT` | Bulk quotes |
| `GET` | `/api/stocks/history?symbol=AAPL&period=1mo` | Historical price data |

---

### 🔊 3. Acoustic Signal Processing Lab — `/acoustic-lab`

Three sub-modules for acoustic signal analysis:

#### 🚗 Doppler Effect Simulator
Generate realistic vehicle-passing sounds using the Doppler formula. Adjustable parameters:
- Vehicle speed (5–150 m/s)
- Horn frequency (100–2000 Hz)
- Pass-by duration & closest distance
- Real-time frequency visualization on canvas

#### 📊 Vehicle Speed Estimator
Upload a `.wav` or `.mp3` recording of a vehicle passing by to:
- Generate a spectrogram via FFT
- Extract the Doppler frequency curve
- Estimate the vehicle's speed and horn frequency
- Configurable frequency band (min/max)

#### 🛸 Drone Sound Detector
Upload audio or use your **live microphone** to detect drone presence:
- Spectral analysis targeting rotor harmonics (80–500 Hz)
- Detection confidence score
- Dominant frequency & harmonic identification
- Signal-to-Noise Ratio (SNR) measurement
- Adjustable detection threshold and frequency band

---

### 🧬 4. Microbiome Signals — `/microbiome`

Visualize gut microbiome abundance data and estimate patient health profiles.

**How to use:** Upload a CSV file containing microbiome data (e.g., the provided `hmp_gut_microbiome.csv` from Desktop or your own dataset).

**Expected CSV columns:** `SampleID, PatientID, Age, Sex, BMI, BodySite, Diagnosis, Bacteroides, Firmicutes, Proteobacteria, Actinobacteria, Fusobacteria, Verrucomicrobia, Tenericutes, Cyanobacteria, Spirochaetes, Synergistetes`

**Visualizations:**
- **Abundance Bar Chart** — bacterial abundances per sample
- **Heatmap** — samples × bacteria abundance matrix
- **Composition Pie** — relative bacterial composition
- **Diversity Plot** — Shannon diversity across samples

**Patient Profile Estimator:**
Select a patient from the dropdown to see:
- Disease risk assessment based on microbiome signature
- Known microbiome–disease associations (IBD, T2D, Obesity, CRC)
- Comparison of patient's profile against the population

---

## 👥 Team

**Team 15** — SBEG205, Spring 2026

---

## 📄 License

This project is developed for academic purposes as part of the Biomedical Engineering curriculum.
