from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
load_dotenv()  # loads .env file so GEMINI_API_KEY is available via os.environ
import numpy as np
import os
from tempfile import NamedTemporaryFile
import pandas as pd
from io import StringIO
import pyedflib
import wfdb
import tempfile
import shutil
import yfinance as yf
# AI and ML classifiers
from models.ecg_classifier import ECGClassifier
from models.eeg_classifier import EEGClassifier
from models.ml_detector import classify_ecg_binary, classify_eeg_classic
from models.predict import predict_bp

DEF_SR = 1000

app = Flask(__name__) 
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.register_blueprint(predict_bp)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock-dashboard')
def stock_dashboard():
    return render_template('stock-dashboard.html')

@app.route('/acoustic-lab')
def acoustic_lab():
    return render_template('acoustic-lab.html')

@app.route('/microbiome')
def microbiome():
    return render_template('microbiome.html')  # serves the new React microbiome dashboard


@app.route('/api/microbiome/summary', methods=['POST'])
def microbiome_summary():
    """Proxy the AI summary request to Gemini so the API key stays server-side."""
    import requests as req_lib

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return jsonify({'success': False, 'error': 'GEMINI_API_KEY not set on server'})

    body = request.get_json()
    if not body or 'prompt' not in body:
        return jsonify({'success': False, 'error': 'Missing prompt'})

    try:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'
        resp = req_lib.post(
            url,
            headers={'Content-Type': 'application/json'},
            json={
                'contents': [{'parts': [{'text': body['prompt']}]}],
                'generationConfig': {'maxOutputTokens': 1000, 'temperature': 0.7}
            },
            timeout=60,
        )
        result = resp.json()

        # Surface Gemini errors clearly instead of crashing on missing keys
        if 'error' in result:
            return jsonify({'success': False, 'error': f"Gemini: {result['error'].get('message', str(result['error']))}"})

        if 'candidates' not in result:
            return jsonify({'success': False, 'error': f"Unexpected response: {str(result)[:300]}"})

        text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'success': True, 'summary': text})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
            ml_result = classify_ecg_binary(signals, sr)
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

"""
stock_routes.py — Real stock data routes using yfinance.

Add these routes to your app.py, and add this to your imports:
    import yfinance as yf

Install dependency:
    pip install yfinance

Then paste the three routes below into app.py (before the `if __name__ == '__main__':` line).
"""

import yfinance as yf
from flask import request, jsonify


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE 1: Single real-time quote
#  GET /api/stocks/quote?symbol=AAPL
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/stocks/quote', methods=['GET'])
def stock_quote():
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'success': False, 'error': 'No symbol provided'})

    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info   # dict with lots of metadata

        # fast_info is lighter-weight but info has more fields
        price      = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose') or price
        change     = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return jsonify({
            'success':    True,
            'symbol':     symbol,
            'name':       info.get('longName') or info.get('shortName') or symbol,
            'price':      round(float(price), 4),
            'change':     round(float(change), 4),
            'change_pct': round(float(change_pct), 4),
            'market_cap': info.get('marketCap'),
            'volume':     info.get('volume') or info.get('regularMarketVolume'),
            'currency':   info.get('currency', 'USD'),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE 2: Bulk quotes (many symbols at once — much faster than looping)
#  GET /api/stocks/bulk?symbols=AAPL,MSFT,TSLA
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/stocks/bulk', methods=['GET'])
def stock_bulk():
    symbols_param = request.args.get('symbols', '')
    if not symbols_param:
        return jsonify({'success': False, 'error': 'No symbols provided'})

    symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()]
    if len(symbols) > 50:
        symbols = symbols[:50]   # cap to avoid abuse

    try:
        # yf.download with group_by='ticker' fetches all in one HTTP request
        # For quotes specifically, Tickers().history is the fastest bulk method
        tickers = yf.Tickers(' '.join(symbols))

        quotes = {}
        for symbol in symbols:
            try:
                t          = tickers.tickers[symbol]
                info       = t.fast_info           # lightweight, no full info dict
                price      = float(info.last_price or 0)
                prev_close = float(info.previous_close or price)
                change     = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0

                quotes[symbol] = {
                    'price':      round(price, 4),
                    'change':     round(change, 4),
                    'change_pct': round(change_pct, 4),
                }
            except Exception:
                # If one ticker fails, return nulls so the UI can show '—'
                quotes[symbol] = {'price': None, 'change': None, 'change_pct': None}

        return jsonify({'success': True, 'quotes': quotes})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE 3: Historical price data for charting
#  GET /api/stocks/history?symbol=AAPL&period=1mo
#
#  period values (yfinance):  5d  1mo  3mo  1y
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/stocks/history', methods=['GET'])
def stock_history():
    symbol = request.args.get('symbol', '').upper().strip()
    period = request.args.get('period', '1mo')

    # Whitelist allowed periods to prevent injection
    allowed_periods = {'5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'}
    if period not in allowed_periods:
        period = '1mo'

    if not symbol:
        return jsonify({'success': False, 'error': 'No symbol provided'})

    # Choose interval based on period
    interval_map = {
        '5d':  '15m',   # 15-minute bars for 1-week view
        '1mo': '1d',    # daily bars for 1-month view
        '3mo': '1d',    # daily bars for 3-month view
        '6mo': '1d',
        '1y':  '1wk',   # weekly bars for 1-year view (less noise)
        '2y':  '1wk',
        '5y':  '1mo',
    }
    interval = interval_map.get(period, '1d')

    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=period, interval=interval)

        if hist.empty:
            return jsonify({'success': False, 'error': f'No data returned for {symbol}'})

        # Format dates for chart labels
        if interval in ('15m', '30m', '1h'):
            labels = [str(ts.strftime('%b %d %H:%M')) for ts in hist.index]
        elif interval in ('1d',):
            labels = [str(ts.strftime('%b %d')) for ts in hist.index]
        else:
            labels = [str(ts.strftime("%b '%y")) for ts in hist.index]

        prices = [round(float(p), 4) for p in hist['Close'].tolist()]

        return jsonify({
            'success': True,
            'symbol':  symbol,
            'period':  period,
            'labels':  labels,
            'prices':  prices,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True, port=5000)