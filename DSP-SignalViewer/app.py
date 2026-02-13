from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tempfile import NamedTemporaryFile
import pandas as pd
from io import StringIO
import pyedflib

DEF_SR = 1000

app = Flask(__name__) 
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Store uploaded data in memory (for simplicity)
signal_data = {}

# signal_data = {
#     'current': {
#         'signals': {'Channel_1': [0.1, 0.2, ...], 'Channel_2': [...]},
#         'sr': 500.0,           # sample rate
#         'channels': ['Channel_1', 'Channel_2'],
#         'time': [0.0, 0.002, 0.004, ...]
#     }
# }

def read_csv_data(file_content):

    df = pd.read_csv(StringIO(file_content))
    
    # Find time column
    time_col = None
    for col in df.columns:
        if 'time' in col.lower() or col.lower() == 't':
            time_col = col
            break
    
    if time_col: #here we get the sr and export the time data 
        time_data = df[time_col].tolist() #delete .values
        signal_cols = [c for c in df.columns if c != time_col]
        sr = 1.0 / np.median(np.diff(time_data)) if len(time_data) > 1 else DEF_SR
    else: # in case there is no time col
        signal_cols = list(df.columns)
        sr = DEF_SR
        time_data = (np.arange(len(df)) / sr).tolist()
    
    signals = {col: df[col].values.tolist() for col in signal_cols} # geting each signal values and put them in a dic.
    
    return signals, float(sr), signal_cols, time_data


def read_edf_data(file_bytes):    
    
    tmp = NamedTemporaryFile(delete=False, suffix='.edf') #delete=False , so it doesn't be deleted after closing it
    tmp.write(file_bytes)
    tmp_path = tmp.name
    
    try:
        edf = pyedflib.EdfReader(tmp_path)
        n_channels = edf.signals_in_file
        channel_names = edf.getSignalLabels()
        sr = edf.getSampleFrequency(0) # get the SR at the col of index 0 
        
        signals = {}
        for i in range(n_channels):
            sig = edf.readSignal(i)
            signals[channel_names[i]] = sig.tolist()
        
        n_samples = len(signals[channel_names[0]])
        time_data = (np.arange(n_samples) / sr).tolist()
        
        edf.close()
        return signals, float(sr), channel_names, time_data
    finally:
        os.unlink(tmp_path) # delete the temp file



@app.route('/') #http://127.0.0.1:5000/ ,  the / in the end refer to the main page 
def index():
    return render_template('index.html') #call the file of html that will show the web page

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Handle file upload
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


@app.route('/api/all_data', methods=['GET'])
def get_all_data():
    # Get all signal data
    if 'current' not in signal_data:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    return jsonify(signal_data['current'])


if __name__ == '__main__':
    app.run(debug=True, port=5000)
