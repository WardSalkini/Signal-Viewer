from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from tempfile import NamedTemporaryFile
import pandas as pd
from io import StringIO
import pyedflib
import wfdb
import tempfile
import shutil

# AI and ML classifiers
from models.ecg_classifier import ECGClassifier
from models.eeg_classifier import EEGClassifier
from models.ml_detector import classify_ecg_classic, classify_eeg_classic

DEF_SR = 1000

app = Flask(__name__) 
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Store uploaded data per window (keyed by window_id)
signal_data = {}

# Initialize AI classifiers
ecg_classifier = ECGClassifier()
eeg_classifier = EEGClassifier()

# signal_data = {
#     'win_0': {
#         'signals': {'Channel_1': [0.1, 0.2, ...], 'Channel_2': [...]},
#         'sr': 500.0,           # sample rate
#         'channels': ['Channel_1', 'Channel_2'],
#         'time': [0.0, 0.002, 0.004, ...],
#         'signal_type': 'ecg'
#     },
#     'win_1': { ... }  # another window's data
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
    tmp.close()  # Must close before pyedflib can open it (Windows locks open files)
    
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
        try:
            os.unlink(tmp_path) # delete the temp file
        except PermissionError:
            pass  # Windows may still hold a lock briefly; file will be cleaned up later


def read_dat_data(file_bytes, filename, all_files):
    """
    Read WFDB .dat file (PhysioNet format).
    If .hea companion file is present → use wfdb library.
    If only .dat is uploaded → read as raw 16-bit binary data.
    If only .hea is uploaded → parse header for channel names, then read .dat raw.
    """
    # Determine record name (strip both .dat and .hea extensions)
    basename = filename
    for ext in ['.dat', '.hea']:
        if basename.lower().endswith(ext):
            basename = basename[:-len(ext)]
            break
    record_name = basename
    
    hea_filename = record_name + '.hea'
    dat_filename = record_name + '.dat'
    
    # Find the actual filenames in all_files (case-insensitive)
    hea_key = None
    dat_key = None
    for fn in all_files.keys():
        if fn.lower() == hea_filename.lower():
            hea_key = fn
        if fn.lower() == dat_filename.lower():
            dat_key = fn
    
    # If both .dat and .hea exist → try full WFDB mode
    if dat_key and hea_key:
        tmp_dir = tempfile.mkdtemp()
        try:
            for fname, fbytes in all_files.items():
                fpath = os.path.join(tmp_dir, os.path.basename(fname))
                with open(fpath, 'wb') as f:
                    f.write(fbytes)
            
            record = wfdb.rdrecord(os.path.join(tmp_dir, record_name))
            sr = float(record.fs)
            channel_names = record.sig_name
            n_samples = record.sig_len
            
            signals = {}
            for i, ch_name in enumerate(channel_names):
                signals[ch_name] = record.p_signal[:, i].tolist()
            
            time_data = (np.arange(n_samples) / sr).tolist()
            return signals, sr, channel_names, time_data
        except Exception:
            # If WFDB fails, fall through to raw mode
            pass
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Parse .hea for channel names and sample rate (if available)
    parsed_sr = DEF_SR
    parsed_channels = None
    n_channels_from_hea = None
    
    if hea_key:
        try:
            hea_text = all_files[hea_key].decode('utf-8', errors='ignore')
            lines = [l.strip() for l in hea_text.split('\n') if l.strip() and not l.strip().startswith('#')]
            if lines:
                # First line: record_name n_channels sample_rate n_samples
                header_parts = lines[0].split()
                if len(header_parts) >= 3:
                    n_channels_from_hea = int(header_parts[1])
                    parsed_sr = float(header_parts[2])
                
                # Following lines: signal descriptions (one per channel)
                parsed_channels = []
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 9:
                        # Last field is usually the channel name/description
                        parsed_channels.append(parts[-1])
                    elif len(parts) >= 1:
                        parsed_channels.append(parts[0])
                
                if n_channels_from_hea and len(parsed_channels) < n_channels_from_hea:
                    parsed_channels = None  # incomplete, ignore
        except Exception:
            pass
    
    # Raw binary mode
    dat_bytes = all_files.get(dat_key, file_bytes) if dat_key else file_bytes
    sr = parsed_sr
    
    raw = np.frombuffer(dat_bytes, dtype=np.int16)
    n_total = len(raw)
    
    # Use channel count from .hea or guess
    if n_channels_from_hea and n_total % n_channels_from_hea == 0:
        n_channels = n_channels_from_hea
    else:
        n_channels = 1
        for nc in [12, 3, 2]:
            if n_total % nc == 0 and n_total // nc > 100:
                n_channels = nc
                break
    
    n_samples = n_total // n_channels
    data = raw[:n_samples * n_channels].reshape(n_samples, n_channels)
    data = data.astype(np.float64) / 200.0
    
    # Use channel names from .hea or generate defaults
    if parsed_channels and len(parsed_channels) == n_channels:
        channel_names = parsed_channels
    else:
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    
    signals = {}
    for i, ch_name in enumerate(channel_names):
        signals[ch_name] = data[:, i].tolist()
    
    time_data = (np.arange(n_samples) / sr).tolist()
    return signals, sr, channel_names, time_data



@app.route('/') #http://127.0.0.1:5000/ ,  the / in the end refer to the main page 
def index():
    return render_template('index.html') #call the file of html that will show the web page

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Handle file upload
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file'})

    # Get window ID (each window has its own data)
    window_id = request.form.get('window_id', '0')

    # Collect all uploaded files (for WFDB which needs .dat + .hea)
    uploaded_files = request.files.getlist('file')
    if not uploaded_files:
        return jsonify({'success': False, 'error': 'No file'})
    
    # Read all files into memory
    all_files = {}
    primary_file = None
    primary_filename = None
    
    for f in uploaded_files:
        fname = f.filename
        fbytes = f.read()
        all_files[fname] = fbytes
        if primary_file is None:
            primary_file = fbytes
            primary_filename = fname
        # Prefer .dat/.csv/.edf as primary, also accept .hea as fallback
        if fname.lower().endswith(('.dat', '.csv', '.edf')):
            primary_file = fbytes
            primary_filename = fname
    
    filename = primary_filename.lower()
    
    try:
        if filename.endswith('.csv'):
            content = primary_file.decode('utf-8')
            signals, sr, channels, time_data = read_csv_data(content)
        elif filename.endswith('.edf'):
            signals, sr, channels, time_data = read_edf_data(primary_file)
        elif filename.endswith('.dat') or filename.endswith('.hea'):
            signals, sr, channels, time_data = read_dat_data(
                primary_file, primary_filename, all_files
            )
        else:
            return jsonify({'success': False, 'error': 'Unsupported format. Use .csv, .edf, or .dat'})
        
        # Auto-detect signal type based on channel names
        ecg_keywords = ['ecg', 'lead', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'i', 'ii', 'iii']
        eeg_keywords = ['eeg', 'fp1', 'fp2', 'f3', 'f4', 'c3', 'c4', 'p3', 'p4', 'o1', 'o2', 'f7', 'f8', 't3', 't4', 'fz', 'cz', 'pz']
        
        ch_lower = [c.lower().strip() for c in channels]
        is_eeg = any(any(kw in ch for kw in eeg_keywords) for ch in ch_lower)
        signal_type = 'eeg' if is_eeg else 'ecg'
        
        # Store per window
        signal_data[window_id] = {
            'signals': signals,
            'sr': sr,
            'channels': channels,
            'time': time_data,
            'signal_type': signal_type
        }
        
        return jsonify({
            'success': True,
            'channels': channels,
            'sr': sr,
            'duration': len(time_data) / sr,
            'n_samples': len(time_data),
            'signal_type': signal_type
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/all_data', methods=['GET'])
def get_all_data():
    window_id = request.args.get('window_id', '0')
    if window_id not in signal_data:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    return jsonify(signal_data[window_id])


@app.route('/api/set_signal_type', methods=['GET'])
def set_signal_type():
    """Allow user to override auto-detected signal type (ecg/eeg)."""
    window_id = request.args.get('window_id', '0')
    signal_type = request.args.get('signal_type', 'ecg')
    if window_id in signal_data:
        signal_data[window_id]['signal_type'] = signal_type
    return jsonify({'success': True, 'signal_type': signal_type})


@app.route('/api/classify', methods=['GET'])
def classify_signal():
    """Run both AI and Classic ML classification on loaded data."""
    window_id = request.args.get('window_id', '0')
    if window_id not in signal_data:
        return jsonify({'success': False, 'error': 'No data loaded'})
    
    data = signal_data[window_id]
    signals = data['signals']
    sr = data['sr']
    signal_type = data.get('signal_type', 'ecg')
    
    try:
        if signal_type == 'ecg':
            ai_result = ecg_classifier.classify(signals, sr)
            ml_result = classify_ecg_classic(signals, sr)
        else:
            ai_result = eeg_classifier.classify(signals, sr)
            ml_result = classify_eeg_classic(signals, sr)
        
        return jsonify({
            'success': True,
            'signal_type': signal_type,
            'ai': ai_result,
            'ml': ml_result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)