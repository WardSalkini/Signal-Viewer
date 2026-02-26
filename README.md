# ECG/EEG Signal Viewer 📊

A web-based signal viewer for ECG and EEG data with smooth real-time animation, multiple viewing modes, and interactive controls.

![Signal Viewer](https://img.shields.io/badge/Python-Flask-blue) ![JavaScript](https://img.shields.io/badge/JavaScript-Chart.js-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## Features ✨https://www.kaggle.com/code/zyadhamed/ecg-ml-dsp-t1

- **Multi-Window Support**: Up to 4 independent signal windows for comparison
- **Smooth Animation**: Real-time playback using `requestAnimationFrame`
- **Multiple Plot Types**:
  - Signal vs Time (Cartesian)
  - Channel vs Channel (XY Plot)
  - Polar Plot (r = signal, θ = time)
  - Polar Ratio Plot (|Ch1| / |Ch2|)
- **Interactive Zoom & Pan**: 
  - Mouse wheel zoom
  - Drag to pan
  - Axis-specific zoom (X, Y, or XY)
- **Playback Controls**: Play, Pause, Reset, Speed control (0.1x - 5x)
- **File Support**: EDF and CSV formats

## Installation 🛠️

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone or download this repository:
```bash
cd DSP-SignalViewer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Start the server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Load data:
   - Click **Demo** for sample ECG data
   - Or upload your own `.edf` or `.csv` file

4. Use the controls:
   - **Plot Type**: Select visualization mode
   - **Channels**: Select which channels to display
   - **Speed**: Adjust playback speed (0.1x - 5x)
   - **Window**: Adjust the time window size
   - **Zoom**: Use +/−/↺ buttons or mouse wheel
   - **Axis Lock**: Click X, Y, or XY to lock zoom/pan to specific axis

## File Formats 📁

### CSV Format
```csv
time,channel1,channel2,channel3
0.000,0.12,0.34,0.56
0.002,0.13,0.35,0.57
...
```

### EDF Format
Standard European Data Format (EDF/EDF+) files are supported.

## Project Structure 📂

```
DSP-SignalViewer/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Frontend UI with JavaScript
└── utils/
    ├── __init__.py
    ├── file_reader.py     # File parsing utilities
    └── plots.py           # Plot generation (legacy Streamlit)
```

## Technologies Used 🔧

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js with Zoom plugin
- **File Parsing**: pyedflib, pandas, numpy

## Controls Reference 🎮

| Control | Description |
|---------|-------------|
| ▶ Play | Start animation |
| ⏸ Pause | Stop animation |
| ↺ Reset | Reset to beginning |
| + / − | Zoom in/out |
| XY / X / Y | Axis-specific zoom mode |
| Position Slider | Navigate through signal |
| Window Slider | Adjust visible time range |
| Speed Slider | Adjust playback speed |

## Screenshots 📸

### Signal vs Time
- Display multiple channels over time
- Color-coded channel differentiation

### Channel vs Channel (XY Plot)
- Plot one channel against another
- Color gradient shows time progression

### Polar Plot
- Signal amplitude mapped to radius
- Time mapped to angle (θ)
- Multiple channels with reference circle

### Polar Ratio
- Ratio of channel magnitudes (|Ch1|/|Ch2|)
- Reference circles at r=1, 2, 3

## License 📄

MIT License - Feel free to use and modify for your projects.

## Authors 👥

Created for DSP Signal Processing course.

---

**Enjoy visualizing your signals! 🎉**
