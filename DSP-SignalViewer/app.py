"""
ECG/EEG Signal Viewer - Flask Backend
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import os
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Store uploaded data in memory (for simplicity)
signal_data = {}


def read_csv_data(file_content):
    """Parse CSV file content."""
    import pandas as pd
    from io import StringIO
    
    df = pd.read_csv(StringIO(file_content))
    
    # Find time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or col.lower() == 't':
            time_col = col
            break
    
    if time_col:
        time_data = df[time_col].values.tolist()
        signal_cols = [c for c in df.columns if c != time_col]
        sr = 1.0 / np.median(np.diff(time_data)) if len(time_data) > 1 else 1000
    else:
        signal_cols = list(df.columns)
        sr = 1000
        time_data = (np.arange(len(df)) / sr).tolist()
    
    signals = {col: df[col].values.tolist() for col in signal_cols}
    
    return signals, float(sr), signal_cols, time_data


def read_edf_data(file_bytes):
    """Parse EDF file."""
    import pyedflib
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        edf = pyedflib.EdfReader(tmp_path)
        n_channels = edf.signals_in_file
        channel_names = edf.getSignalLabels()
        sr = edf.getSampleFrequency(0)
        
        signals = {}
        for i in range(n_channels):
            sig = edf.readSignal(i)
            signals[channel_names[i]] = sig.tolist()
        
        n_samples = len(signals[channel_names[0]])
        time_data = (np.arange(n_samples) / sr).tolist()
        
        edf.close()
        return signals, float(sr), channel_names, time_data
    finally:
        os.unlink(tmp_path)


def generate_sample_data(duration=20.0, sr=500.0, n_channels=3):
    """Generate sample ECG-like data."""
    n_samples = int(duration * sr)
    t = np.linspace(0, duration, n_samples)
    
    signals = {}
    channel_names = []
    
    for i in range(n_channels):
        name = f"Channel_{i+1}"
        channel_names.append(name)
        
        freq = 1.0 + i * 0.2
        signal = np.zeros(n_samples)
        signal += 0.1 * np.sin(2 * np.pi * 0.15 * t)
        
        beat_interval = 1.0 / freq
        beat_times = np.arange(0, duration, beat_interval)
        
        for bt in beat_times:
            idx = int(bt * sr)
            if idx < n_samples - 50:
                signal[idx:idx+10] += 0.2 * np.sin(np.linspace(0, np.pi, 10))
                signal[idx+15:idx+20] -= 0.3
                signal[idx+20:idx+25] += 1.5
                signal[idx+25:idx+30] -= 0.5
                signal[idx+35:idx+50] += 0.3 * np.sin(np.linspace(0, np.pi, 15))
        
        signal += 0.05 * np.random.randn(n_samples)
        signals[name] = signal.tolist()
    
    return signals, sr, channel_names, t.tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sample', methods=['GET'])
def get_sample():
    """Get sample data."""
    signals, sr, channels, time_data = generate_sample_data()
    signal_data['current'] = {
        'signals': signals,
        'sr': sr,
        'channels': channels,
        'time': time_data
    }
    return jsonify({
        'success': True,
        'channels': channels,
        'sr': sr,
        'duration': len(time_data) / sr,
        'n_samples': len(time_data)
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})
    
    file = request.files['file']
    filename = file.filename.lower()
    
    try:
        if filename.endswith('.csv'):
            content = file.read().decode('utf-8')
            signals, sr, channels, time_data = read_csv_data(content)
        elif filename.endswith('.edf'):
            content = file.read()
            signals, sr, channels, time_data = read_edf_data(content)
        else:
            return jsonify({'success': False, 'error': 'Unsupported format'})
        
        signal_data['current'] = {
            'signals': signals,
            'sr': sr,
            'channels': channels,
            'time': time_data
        }
        
        return jsonify({
            'success': True,
            'channels': channels,
            'sr': sr,
            'duration': len(time_data) / sr,
            'n_samples': len(time_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/data', methods=['GET'])
def get_data():
    """Get signal data for a time range."""
    if 'current' not in signal_data:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    data = signal_data['current']
    start = int(request.args.get('start', 0))
    end = int(request.args.get('end', len(data['time'])))
    channels = request.args.get('channels', ','.join(data['channels'])).split(',')
    
    result = {
        'time': data['time'][start:end],
        'signals': {ch: data['signals'][ch][start:end] for ch in channels if ch in data['signals']},
        'sr': data['sr']
    }
    
    return jsonify(result)


@app.route('/api/all_data', methods=['GET'])
def get_all_data():
    """Get all signal data."""
    if 'current' not in signal_data:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    return jsonify(signal_data['current'])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
