"""
Combined ECG / EEG Arrhythmia & Anomaly Detector
═══════════════════════════════════════════════════════════════════════════════

MODE A — Binary ECG classifier  (Kaggle CPSC-2018 pipeline)
    Labels  : Normal  |  Arrhythmia (AF / AFL)
    Input   : 12-lead .mat files  (SNOMED codes 426783006 / 164889003 / 164890007)
    Pipeline: load_labels → build_dataset → train_model → save_model → predict_mat

MODE B — Multi-class ECG classifier  (Classic ML v3)
    Labels  : NORMAL | STACH | SBRAD | IRBBB | CRBBB
    Input   : signals_dict  {channel_name: np.ndarray}
    Entry   : classify_ecg_classic(signals_dict, sr)

MODE C — Multi-class EEG classifier  (Classic ML v3)
    Labels  : NORM | SEIZ | SLOW | SPIKE | ART
    Input   : signals_dict  {channel_name: np.ndarray}
    Entry   : classify_eeg_classic(signals_dict, sr)

Modes B and C share all DSP utilities (bandpass, R-peak, wavelets, HRV, etc.)
Mode A shares the same bandpass and R-peak code, ensuring consistency.
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

HEA_DIR = '/kaggle/input/china-physiological-signal-challenge-in-2018/Training_WFDB'
FS      = 500
LEADS   = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# Binary SNOMED map  →  0 = Normal, 1 = Arrhythmia
SNOMED_MAP = {
    '426783006': 0,   # Sinus Rhythm
    '164889003': 1,   # Atrial Fibrillation
    '164890007': 1,   # Atrial Flutter
}
BINARY_LABEL_NAMES = {0: 'Normal', 1: 'Arrhythmia (AF/AFL)'}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DSP UTILITIES  (used by ALL three modes)
# ══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(sig, sr, low_hz=0.5, high_hz=40.0, order=4):
    """
    Zero-phase Butterworth bandpass.
    Default 0.5–40 Hz matches Kaggle notebook exactly:
        butter(4, [0.5, 40], btype='bandpass', fs=fs)
    """
    nyq = sr / 2.0
    lo  = np.clip(low_hz  / nyq, 1e-4, 0.9999)
    hi  = np.clip(high_hz / nyq, 1e-4, 0.9999)
    if lo >= hi:
        return sig.copy()
    b, a = scipy_signal.butter(order, [lo, hi], btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def compute_autocorrelation_score(sig):
    """Periodicity score 0–1 via normalised autocorrelation peak."""
    sig  = sig - np.mean(sig)
    norm = np.sum(sig ** 2)
    if norm < 1e-10:
        return 0.0
    ac = np.correlate(sig, sig, mode='full')
    ac = ac[len(ac) // 2:] / norm
    peaks, _ = scipy_signal.find_peaks(ac[1:], height=0.1)
    return float(ac[peaks[0] + 1]) if len(peaks) else 0.0


def compute_wavelet_features(sig, wavelet='db4', levels=5):
    """DWT energy ratios per sub-band (db4, 5 levels)."""
    if not _HAS_PYWT or len(sig) < 2 ** levels:
        return {}
    coeffs = pywt.wavedec(sig, wavelet, level=levels)
    names  = [f'cA{levels}'] + [f'cD{i}' for i in range(levels, 0, -1)]
    total  = sum(np.sum(c ** 2) for c in coeffs) + 1e-12
    return {
        f'{n}_energy_ratio': float(np.sum(c ** 2) / total)
        for n, c in zip(names, coeffs)
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED ECG UTILITIES  (R-peaks, HRV, morphology)
# ══════════════════════════════════════════════════════════════════════════════

def detect_r_peaks(sig, sr):
    """
    Improved Pan-Tompkins R-peak detector.

    Reference (Kaggle notebook):
        diff_sig   = np.diff(signal)
        squared    = diff_sig ** 2
        integrated = convolve(squared, ones(win)/win, 'same')
        peaks, _   = find_peaks(integrated,
                                 distance=int(0.2*fs),
                                 height=np.mean(integrated))

    Improvements over notebook:
      • Bandpass 5–15 Hz before diff (isolates QRS energy)
      • Refractory period 0.30 s (vs 0.20 s) — avoids double-detection
      • Adaptive fallback thresholds for weak/noisy signals
      • Refine each peak to true local amplitude max ±50 ms
    """
    sig = np.array(sig, dtype=np.float64)

    filtered   = bandpass_filter(sig, sr, 5.0, 15.0, order=2)
    diff_sig   = np.diff(filtered)
    squared    = diff_sig ** 2
    win        = max(3, int(0.15 * sr))
    integrated = np.convolve(squared, np.ones(win) / win, mode='same')

    min_dist = max(1, int(0.30 * sr))
    peaks    = np.array([], dtype=int)
    for height_frac in [1.0, 0.5, 0.25, 0.10]:
        threshold = height_frac * np.mean(integrated)
        peaks, _  = scipy_signal.find_peaks(
            integrated, height=threshold, distance=min_dist)
        if len(peaks) >= 3:
            break

    if len(peaks) < 3:
        abs_sig = np.abs(sig - np.mean(sig))
        peaks, _ = scipy_signal.find_peaks(
            abs_sig,
            height=np.mean(abs_sig) + 0.5 * np.std(abs_sig),
            distance=min_dist)

    refine  = max(1, int(0.05 * sr))
    refined = []
    for p in peaks:
        s = max(0, p - refine)
        e = min(len(sig), p + refine)
        refined.append(s + int(np.argmax(np.abs(sig[s:e]))))
    return np.array(refined, dtype=int)


def compute_hrv_extended(rr_ms):
    """Poincaré SD1/SD2, SDNN, pNN50, RR entropy."""
    if len(rr_ms) < 3:
        return {k: 0 for k in ('sdnn','pnn50','rr_entropy',
                                'poincare_sd1','poincare_sd2',
                                'poincare_sd1_sd2_ratio')}
    rr   = rr_ms / 1000.0
    rr_d = np.diff(rr_ms)
    sdnn = float(np.std(rr, ddof=1))
    nn50 = int(np.sum(np.abs(rr_d) > 50))
    pnn50 = nn50 / len(rr_d) * 100 if len(rr_d) else 0.0

    counts, _ = np.histogram(rr, bins=min(10, len(rr)))
    probs     = counts / (counts.sum() + 1e-9)
    rr_entropy = float(scipy_entropy(probs + 1e-9))

    rr1, rr2 = rr[:-1], rr[1:]
    sd1 = float(np.std((rr2 - rr1) / np.sqrt(2)))
    sd2 = float(np.std((rr2 + rr1) / np.sqrt(2)))
    return {
        'sdnn':                   sdnn,
        'pnn50':                  pnn50,
        'rr_entropy':             rr_entropy,
        'poincare_sd1':           sd1,
        'poincare_sd2':           sd2,
        'poincare_sd1_sd2_ratio': sd1 / (sd2 + 1e-9),
    }


def estimate_qrs_width(sig, r_peaks, sr):
    """Half-max QRS width in ms. Normal <100 | IRBBB 100–120 | CRBBB >120."""
    if len(r_peaks) < 2:
        return 90.0
    sig    = np.array(sig, dtype=np.float64)
    hw     = max(1, int(0.08 * sr))
    widths = []
    for peak in r_peaks:
        s, e = max(0, peak - hw), min(len(sig), peak + hw)
        seg  = np.abs(sig[s:e])
        if len(seg) < 4:
            continue
        pv = np.max(seg)
        if pv < 1e-10:
            continue
        above = np.where(seg > 0.5 * pv)[0]
        if len(above) >= 2:
            w_ms = (above[-1] - above[0]) / sr * 1000
            if 20 < w_ms < 250:
                widths.append(w_ms)
    return float(np.median(widths)) if widths else 90.0


def compute_beat_template_corr(sig, r_peaks, sr):
    """Mean inter-beat template correlation (morphology consistency)."""
    sig  = np.array(sig, dtype=np.float64)
    hw   = max(1, int(0.30 * sr))
    beats = [sig[p - hw: p + hw]
             for p in r_peaks if p - hw >= 0 and p + hw <= len(sig)]
    if len(beats) < 2:
        return 0.0
    ml  = min(len(b) for b in beats)
    B   = np.array([b[:ml] for b in beats])
    B   = B - B.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    B  /= norm
    C   = np.dot(B, B.T)
    n   = len(B)
    return float((C.sum() - n) / (n * (n - 1) + 1e-9))


def estimate_pr_interval(sig, r_peaks, sr):
    """Rough PR interval in ms from gradient onset before each R peak."""
    sig    = np.array(sig, dtype=np.float64)
    pr_win = max(1, int(0.20 * sr))
    pr_list = []
    for peak in r_peaks:
        s   = max(0, peak - pr_win)
        seg = sig[s:peak]
        if len(seg) > 4:
            d = np.abs(np.gradient(seg))
            cands = np.where(d > 0.3 * np.max(d))[0]
            if len(cands):
                pr_ms = (peak - (s + cands[0])) / sr * 1000
                if 80 < pr_ms < 300:
                    pr_list.append(pr_ms)
    return float(np.median(pr_list)) if pr_list else 160.0


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — BINARY KAGGLE PIPELINE  (Normal vs Arrhythmia)
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(hea_dir=HEA_DIR):
    """Parse .hea files and return a DataFrame with filename + binary target."""
    records = []
    for fname in sorted(os.listdir(hea_dir)):
        if not fname.endswith('.hea'):
            continue
        rec_id   = fname.replace('.hea', '')
        hea_path = os.path.join(hea_dir, fname)
        with open(hea_path, 'r') as f:
            lines = f.readlines()
        dx_codes = []
        for line in lines:
            if line.startswith('#Dx:'):
                dx_codes = [c.strip() for c in line.replace('#Dx:', '').strip().split(',')]
                break
        target = None
        for code in dx_codes:
            if code in SNOMED_MAP:
                target = SNOMED_MAP[code]
                if target == 1:   # arrhythmia takes priority
                    break
        if target is None:
            continue
        records.append({'filename': os.path.join(hea_dir, rec_id), 'target': target})

    df = pd.DataFrame(records)
    print(f"Total records kept : {len(df)}")
    print(f"  Normal           : {(df['target'] == 0).sum()}")
    print(f"  Arrhythmia       : {(df['target'] == 1).sum()}")
    return df


def _extract_lead_features_binary(sig, fs=FS):
    """
    Per-lead features matching the Kaggle notebook exactly.
    Statistical  : mean, std, min, max, range, skew, kurtosis, rms
    RR / HR      : rr_mean, rr_std, rr_min, rr_max, rr_range, heart_rate, rmssd
    """
    sig     = bandpass_filter(np.array(sig, float), fs, 0.5, 40.0)
    r_peaks = detect_r_peaks(sig, fs)
    feats   = {
        'mean':     float(np.mean(sig)),
        'std':      float(np.std(sig)),
        'min':      float(np.min(sig)),
        'max':      float(np.max(sig)),
        'range':    float(np.max(sig) - np.min(sig)),
        'skew':     float(skew(sig)),
        'kurtosis': float(kurtosis(sig)),
        'rms':      float(np.sqrt(np.mean(sig ** 2))),
    }
    if len(r_peaks) >= 3:
        rr = np.diff(r_peaks) / fs * 1000
        feats.update({
            'rr_mean':    float(np.mean(rr)),
            'rr_std':     float(np.std(rr)),
            'rr_min':     float(np.min(rr)),
            'rr_max':     float(np.max(rr)),
            'rr_range':   float(np.max(rr) - np.min(rr)),
            'heart_rate': float(60000.0 / np.mean(rr)),
            'rmssd':      float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        })
    else:
        for k in ('rr_mean','rr_std','rr_min','rr_max','rr_range','heart_rate','rmssd'):
            feats[k] = np.nan
    return feats


def extract_features_12lead(signal_12, fs=FS):
    """Extract per-lead features for all 12 leads and flatten to one dict."""
    all_feats = {}
    for i, lead_name in enumerate(LEADS):
        feats = _extract_lead_features_binary(signal_12[i, :], fs)
        for k, v in feats.items():
            all_feats[f'{lead_name}_{k}'] = v
    return all_feats


def build_dataset(df):
    """Load .mat files, extract features, return (X DataFrame, y array)."""
    rows, targets = [], []
    for _, row in df.iterrows():
        mat_path = row['filename'] + '.mat'
        if not os.path.exists(mat_path):
            continue
        mat       = scipy.io.loadmat(mat_path)
        signal_12 = mat['val'].astype(float)
        if signal_12.ndim != 2 or signal_12.shape[0] != 12:
            print(f"Unexpected shape {signal_12.shape} — skipping")
            continue
        rows.append(extract_features_12lead(signal_12))
        targets.append(row['target'])

    X = pd.DataFrame(rows)
    y = np.array(targets)
    print(f"Dataset built: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Normal     : {(y == 0).sum()}")
    print(f"  Arrhythmia : {(y == 1).sum()}")
    return X, y


def train_model(X, y):
    """Train XGBoost binary classifier and print evaluation metrics."""
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, stratify=y, random_state=42)

    neg, pos     = np.bincount(y_train)
    scale_pos_wt = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_wt,
        eval_metric='logloss',
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('\n── Classification Report ──')
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Arrhythmia']))
    print(f'ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    return model


def save_model(model, feature_names, save_dir='./model'):
    """Persist XGBoost model + feature names to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_model(os.path.join(save_dir, 'xgb_model.json'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.joblib'))
    print(f'Model saved to {save_dir}/')


def load_model(model_dir='./model'):
    """Load XGBoost model + feature names from disk."""
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'xgb_model.json'))
    feature_names = joblib.load(os.path.join(model_dir, 'feature_names.joblib'))
    return model, feature_names


def predict_mat(mat_path, model_dir='./model'):
    """
    Run binary inference on a single .mat ECG file.

    Returns
    -------
    dict with keys: predicted_class, predicted_label,
                    probability_normal, probability_arrhythmia
    """
    model, feature_names = load_model(model_dir)

    mat       = scipy.io.loadmat(mat_path)
    signal_12 = mat['val'].astype(float)
    if signal_12.ndim != 2 or signal_12.shape[0] != 12:
        raise ValueError(f'Expected shape (12, n_samples), got {signal_12.shape}')

    feats = extract_features_12lead(signal_12)
    X     = pd.DataFrame([feats])[feature_names].fillna(0)

    pred_class = model.predict(X.values)[0]
    pred_proba = model.predict_proba(X.values)[0]

    result = {
        'predicted_class':        int(pred_class),
        'predicted_label':        BINARY_LABEL_NAMES[int(pred_class)],
        'probability_normal':     round(float(pred_proba[0]), 4),
        'probability_arrhythmia': round(float(pred_proba[1]), 4),
    }
    print(f"Prediction  : {result['predicted_label']}")
    print(f"  Normal     : {result['probability_normal']:.2%}")
    print(f"  Arrhythmia : {result['probability_arrhythmia']:.2%}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — MULTI-CLASS ECG CLASSIFIER  (Classic ML v3)
# ══════════════════════════════════════════════════════════════════════════════

ECG_PATHOLOGY_NAMES = {
    'NORMAL': 'Normal Sinus Rhythm',
    'STACH':  'Sinus Tachycardia',
    'SBRAD':  'Sinus Bradycardia',
    'IRBBB':  'Incomplete Right Bundle Branch Block',
    'CRBBB':  'Complete Right Bundle Branch Block',
}


def _extract_lead_features_multiclass(sig, sr):
    """
    Same 15 notebook features as the binary pipeline, but returns
    both the feature dict AND the detected R-peaks (needed by Mode B).
    """
    sig     = bandpass_filter(np.array(sig, float), sr, 0.5, 40.0)
    r_peaks = detect_r_peaks(sig, sr)
    feats   = {
        'mean':     float(np.mean(sig)),
        'std':      float(np.std(sig)),
        'min':      float(np.min(sig)),
        'max':      float(np.max(sig)),
        'range':    float(np.max(sig) - np.min(sig)),
        'skew':     float(skew(sig)),
        'kurtosis': float(kurtosis(sig)),
        'rms':      float(np.sqrt(np.mean(sig ** 2))),
    }
    if len(r_peaks) >= 3:
        rr = np.diff(r_peaks) / sr * 1000
        feats.update({
            'rr_mean':    float(np.mean(rr)),
            'rr_std':     float(np.std(rr)),
            'rr_min':     float(np.min(rr)),
            'rr_max':     float(np.max(rr)),
            'rr_range':   float(np.max(rr) - np.min(rr)),
            'heart_rate': float(60000.0 / np.mean(rr)),
            'rmssd':      float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        })
    else:
        for k in ('rr_mean','rr_std','rr_min','rr_max','rr_range','heart_rate','rmssd'):
            feats[k] = 0.0
    return feats, r_peaks


def classify_ecg_classic(signals_dict, sr):
    """
    Multi-class ECG classification.
    Input : signals_dict — {channel_name: np.ndarray}
    Output: dict with prediction, class_name, confidence, probabilities, details

    Labels : NORMAL | STACH | SBRAD | IRBBB | CRBBB

    Feature layers
    ──────────────
    Layer 1 (Kaggle notebook): mean, std, min, max, range, skew, kurtosis, rms,
                                rr_mean, rr_std, rr_min, rr_max, rr_range,
                                heart_rate, rmssd
    Layer 2 (extended)       : Poincaré SD1/SD2, SDNN, pNN50, RR entropy,
                                QRS width, beat template correlation,
                                PR interval, periodicity, wavelet sub-bands
    """
    channels = list(signals_dict.keys())

    lead_feats_list, all_r_peaks, all_rr_ms = [], [], []
    for ch in channels:
        raw    = np.array(signals_dict[ch], dtype=np.float64)
        lf, rp = _extract_lead_features_multiclass(raw, sr)
        lead_feats_list.append(lf)
        all_r_peaks.append(rp)
        if len(rp) >= 3:
            all_rr_ms.append(np.diff(rp) / sr * 1000)

    def avg_lf(key):
        return float(np.mean([f[key] for f in lead_feats_list]))

    hr          = avg_lf('heart_rate')
    rr_mean_ms  = avg_lf('rr_mean')
    rr_std_ms   = avg_lf('rr_std')
    rr_range_ms = avg_lf('rr_range')
    rmssd_ms    = avg_lf('rmssd')

    primary_sig = bandpass_filter(
        np.array(signals_dict[channels[0]], float), sr, 0.5, 40.0)
    primary_rp  = all_r_peaks[0]

    rr_ms_all  = np.concatenate(all_rr_ms) if all_rr_ms else np.array([])
    hrv_ext    = compute_hrv_extended(rr_ms_all)
    qrs_width  = float(np.mean([
        estimate_qrs_width(
            bandpass_filter(np.array(signals_dict[c], float), sr, 0.5, 40.0),
            all_r_peaks[i], sr)
        for i, c in enumerate(channels)]))
    tmpl_corr  = compute_beat_template_corr(primary_sig, primary_rp, sr)
    pr_ms      = estimate_pr_interval(primary_sig, primary_rp, sr)
    periodicity = compute_autocorrelation_score(primary_sig)
    wavelet    = compute_wavelet_features(primary_sig)
    wt_lo      = (wavelet.get('cD5_energy_ratio', 0) + wavelet.get('cD4_energy_ratio', 0))
    wt_mid     = wavelet.get('cD3_energy_ratio', 0)

    sdnn    = hrv_ext['sdnn']
    pnn50   = hrv_ext['pnn50']
    sd1     = hrv_ext['poincare_sd1']
    sd2     = hrv_ext['poincare_sd2']
    sd1_sd2 = hrv_ext['poincare_sd1_sd2_ratio']

    rr_cv      = (rr_std_ms / (rr_mean_ms + 1e-9)) if rr_mean_ms > 0 else 1.0
    regularity = float(np.clip(1.0 - rr_cv * 4, 0, 1))

    reasons, scores = [], {}

    # Confidence factor: with few peaks, HR/QRS estimates are unreliable
    # Need at least ~10 peaks (8+ seconds) for trustworthy classification
    max_peaks = max(len(rp) for rp in all_r_peaks) if all_r_peaks else 0
    n_samples = len(signals_dict[channels[0]])
    duration_s = n_samples / sr
    # Both peak count AND duration must be sufficient
    peak_conf = float(np.clip((max_peaks - 3) / 10.0, 0.0, 1.0))
    dur_conf  = float(np.clip((duration_s - 2) / 8.0, 0.0, 1.0))
    # Square it for aggressive dampening on short / few-peak signals
    hr_confidence = (peak_conf * dur_conf) ** 2   # very aggressive for HR-based
    # QRS width can be estimated from just 2-3 beats — use gentler dampening
    qrs_confidence = float(np.clip(max_peaks / 5.0, 0.2, 1.0))

    # STACH  (HR-based → use hr_confidence)
    s = 0.0
    if hr >= 100:
        s += min(50 + (hr - 100) * 2.0, 90)
        reasons.append(f"Tachycardia: HR={hr:.0f} bpm")
    elif hr >= 90:
        s += (hr - 90) * 4.0
    if hr >= 100 and regularity > 0.6: s += 8
    if hr >= 100 and tmpl_corr > 0.70: s += 5
    if hr >= 100 and wt_mid > 0.30:    s += 5
    scores['STACH'] = float(np.clip(s * hr_confidence, 0, 95))

    # SBRAD  (HR-based → use hr_confidence)
    s = 0.0
    if 0 < hr < 60:
        s += min(50 + (60 - hr) * 2.5, 90)
        reasons.append(f"Bradycardia: HR={hr:.0f} bpm")
    elif 0 < hr < 65:
        s += (65 - hr) * 7.0
    if 0 < hr < 60 and regularity > 0.5: s += 8
    if pr_ms > 200:
        s += 5
        reasons.append(f"Prolonged PR: {pr_ms:.0f} ms")
    scores['SBRAD'] = float(np.clip(s * hr_confidence, 0, 95))

    # IRBBB  (QRS-based → use qrs_confidence)
    s = 0.0
    if 100 <= qrs_width <= 125:
        s += 30 + (qrs_width - 100) * 2.4
        reasons.append(f"Widened QRS: {qrs_width:.0f} ms (IRBBB range)")
    elif 95 <= qrs_width < 100:
        s += (qrs_width - 95) * 5.0
    if qrs_width > 100 and tmpl_corr < 0.70: s += 8
    if qrs_width > 100 and wt_lo > 0.40:     s += 7
    scores['IRBBB'] = float(np.clip(s * qrs_confidence, 0, 95))

    # CRBBB  (QRS-based → use qrs_confidence)
    s = 0.0
    if qrs_width > 125:
        s += min(50 + (qrs_width - 125) * 2.0, 90)
        reasons.append(f"Wide QRS: {qrs_width:.0f} ms (CRBBB range)")
    elif qrs_width > 118:
        s += (qrs_width - 118) * 8.0
    if qrs_width > 120 and tmpl_corr < 0.65: s += 10
    if qrs_width > 120 and wt_lo > 0.50:     s += 8
    scores['CRBBB'] = float(np.clip(s * qrs_confidence, 0, 95))

    # NORMAL
    max_path = max(scores.values())
    s = 0.0
    if 60 <= hr <= 100:
        s += 45.0
        reasons.append(f"Normal HR: {hr:.0f} bpm")
    elif hr_confidence < 0.5:
        # With few peaks HR is unreliable — don't penalise heavily
        s += 40.0
        reasons.append(f"HR={hr:.0f} bpm (low confidence, only {max_peaks} peaks in {duration_s:.1f}s)")
    if regularity > 0.60:       s += 15.0
    if tmpl_corr > 0.75:
        s += 10.0
        reasons.append(f"Consistent morphology (corr={tmpl_corr:.2f})")
    if 70 <= qrs_width <= 110:  s += 10.0
    if periodicity > 0.30:      s += 8.0
    if sdnn < 0.10:             s += 5.0
    if 120 <= pr_ms <= 200:     s += 5.0
    s *= max(0.10, 1.0 - max_path / 110.0)
    scores['NORMAL'] = float(np.clip(s, 8.0, 90.0))

    for k in scores:
        scores[k] = max(scores[k], 2.0)

    total      = sum(scores.values())
    probs      = {k: round(v / total * 100, 2) for k, v in scores.items()}
    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]

    detected = sorted(
        [(k, v) for k, v in probs.items() if k != 'NORMAL' and v > 5],
        key=lambda x: x[1], reverse=True)

    return {
        'prediction':    prediction,
        'class_name':    ECG_PATHOLOGY_NAMES.get(prediction, prediction),
        'confidence':    round(confidence, 2),
        'probabilities': probs,
        'model': 'Classic ML v3 — ECG (Kaggle features + Poincaré + Wavelet + Beat-Template)',
        'details': {
            'heart_rate':           round(hr, 1),
            'rr_mean_ms':           round(rr_mean_ms, 1),
            'rr_std_ms':            round(rr_std_ms, 1),
            'rr_range_ms':          round(rr_range_ms, 1),
            'rmssd_ms':             round(rmssd_ms, 1),
            'regularity':           round(regularity, 3),
            'sdnn_s':               round(sdnn, 4),
            'pnn50':                round(pnn50, 2),
            'poincare_sd1':         round(sd1, 4),
            'poincare_sd2':         round(sd2, 4),
            'poincare_sd1_sd2':     round(sd1_sd2, 3),
            'qrs_width_ms':         round(qrs_width, 1),
            'pr_ms':                round(pr_ms, 1),
            'template_corr':        round(tmpl_corr, 3),
            'wavelet_cD3_ratio':    round(wavelet.get('cD3_energy_ratio', 0), 3),
            'n_peaks':              len(primary_rp),
            'n_channels':           len(channels),
            'reasons':              reasons,
            'detected_pathologies': [
                {'code': d[0], 'name': ECG_PATHOLOGY_NAMES.get(d[0], d[0]),
                 'confidence': round(d[1], 2)}
                for d in detected],
        }
    }


def classify_ecg_binary(signals_dict, sr):
    """
    Binary ECG classification: Normal vs Abnormal.
    
    Reuses the multi-class feature extraction from classify_ecg_classic,
    then collapses all pathology scores into a single ABNORMAL score.
    
    Returns: dict with NORMAL / ABNORMAL probabilities summing to 100%.
    """
    # Get the full multi-class result (uses all the feature extraction)
    mc = classify_ecg_classic(signals_dict, sr)
    mc_probs = mc.get('probabilities', {})

    # Collapse: NORMAL stays, everything else → ABNORMAL
    normal_pct = mc_probs.get('NORMAL', 50.0)
    abnormal_pct = sum(v for k, v in mc_probs.items() if k != 'NORMAL')

    # Re-normalise so they sum to exactly 100
    total = normal_pct + abnormal_pct or 100.0
    normal_pct = round(normal_pct / total * 100, 2)
    abnormal_pct = round(abnormal_pct / total * 100, 2)

    is_normal = normal_pct >= abnormal_pct

    return {
        'prediction':    'NORMAL' if is_normal else 'ABNORMAL',
        'class_name':    'Normal Sinus Rhythm' if is_normal else 'Abnormal ECG',
        'confidence':    round(normal_pct if is_normal else abnormal_pct, 2),
        'probabilities': {
            'NORMAL':   normal_pct,
            'ABNORMAL': abnormal_pct,
        },
        'model': 'Classic ML — ECG Binary (Kaggle features + Poincaré + Wavelet)',
        'details': mc.get('details', {}),
    }


EEG_BANDS = {
    'delta': (0.5,  4),
    'theta': (4,    8),
    'alpha': (8,   13),
    'beta':  (13,  30),
    'gamma': (30,  50),
}
EEG_LABEL_NAMES = {
    'NORM':  'Normal',
    'SEIZ':  'Seizure',
    'SLOW':  'Slow-wave Abnormality',
    'SPIKE': 'Spike-wave Discharge',
    'ART':   'Artifact Anomaly',
}


def _band_powers(sig, sr):
    nperseg   = min(len(sig), max(64, int(sr * 2)))
    freqs, psd = scipy_signal.welch(sig, fs=sr, nperseg=nperseg)
    total      = np.sum(psd) + 1e-12
    return {b: float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]) / total * 100)
            for b, (lo, hi) in EEG_BANDS.items()}


def _spectral_edge(sig, sr, edge=0.95):
    nperseg   = min(len(sig), max(64, int(sr * 2)))
    freqs, psd = scipy_signal.welch(sig, fs=sr, nperseg=nperseg)
    cum        = np.cumsum(psd) / (np.sum(psd) + 1e-12)
    return float(freqs[min(np.searchsorted(cum, edge), len(freqs) - 1)])


def _spectral_flatness(sig, sr):
    nperseg   = min(len(sig), max(64, int(sr * 2)))
    _, psd     = scipy_signal.welch(sig, fs=sr, nperseg=nperseg)
    psd        = np.abs(psd) + 1e-12
    return float(np.exp(np.mean(np.log(psd))) / np.mean(psd))


def _hjorth(sig):
    d1  = np.diff(sig)
    d2  = np.diff(d1)
    act = float(np.var(sig))
    mob = float(np.sqrt(np.var(d1) / (act + 1e-12)))
    cpx = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mob + 1e-12))
    return act, mob, cpx


def _spike_rate(sig, sr):
    z    = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    sp, _ = scipy_signal.find_peaks(z, height=4.0,
                                     distance=max(1, int(0.02 * sr)))
    return len(sp) / (len(sig) / sr + 1e-9)


def _line_length(sig):
    return float(np.sum(np.abs(np.diff(sig))))


def _sample_entropy(sig, m=2, r_frac=0.2):
    sig = np.array(sig, dtype=np.float64)
    ds  = sig[::max(1, len(sig) // 500)]
    nd  = len(ds)
    r   = r_frac * np.std(ds)
    if nd < m + 2 or r < 1e-10:
        return 0.0
    A, B = 0, 0
    for i in range(nd - m - 1):
        tm  = ds[i:i + m]
        tm1 = ds[i:i + m + 1]
        for j in range(i + 1, nd - m):
            if np.max(np.abs(tm  - ds[j:j + m]))     < r:
                B += 1
                if np.max(np.abs(tm1 - ds[j:j + m + 1])) < r:
                    A += 1
    return float(-np.log((A + 1e-9) / (B + 1e-9)))


def _permutation_entropy(sig, order=3, delay=1):
    from itertools import permutations
    import math
    sig   = np.array(sig, dtype=np.float64)
    n     = len(sig)
    if n < order:
        return 0.0
    perms = {p: 0 for p in permutations(range(order))}
    for i in range(n - (order - 1) * delay):
        pat = tuple(np.argsort([sig[i + j * delay] for j in range(order)]))
        if pat in perms:
            perms[pat] += 1
    counts = np.array([v for v in perms.values() if v > 0], dtype=float)
    probs  = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)) / np.log2(math.factorial(order)))


def _higuchi_fd(sig, k_max=10):
    sig = np.array(sig, dtype=np.float64)
    n   = len(sig)
    if n < k_max * 2:
        return 1.5
    lk = []
    for k in range(1, k_max + 1):
        ls = []
        for m in range(1, k + 1):
            idx = np.arange(m - 1, n, k)
            xm  = sig[idx]
            if len(xm) < 2:
                continue
            ls.append((np.sum(np.abs(np.diff(xm))) * (n - 1)) / (k * len(xm) ** 2))
        if ls:
            lk.append(np.mean(ls))
    if len(lk) < 2:
        return 1.5
    slope, _ = np.polyfit(np.log(np.arange(1, len(lk) + 1)),
                          np.log(np.array(lk) + 1e-12), 1)
    return float(-slope)


def _inter_channel_sync(sigs, sr):
    if len(sigs) < 2:
        return 0.5
    nperseg = min(len(sigs[0]), max(64, int(sr * 2)))
    syncs   = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            f, Cxy = scipy_signal.coherence(sigs[i], sigs[j], fs=sr, nperseg=nperseg)
            mask   = (f >= 4) & (f <= 13)
            if mask.sum():
                syncs.append(float(np.mean(Cxy[mask])))
    return float(np.mean(syncs)) if syncs else 0.5


def classify_eeg_classic(signals_dict, sr):
    """
    Multi-class EEG classification.
    Input : signals_dict — {channel_name: np.ndarray}
    Output: dict with prediction, class_name, confidence, probabilities, details

    Labels : NORM | SEIZ | SLOW | SPIKE | ART
    """
    channels  = list(signals_dict.keys())
    proc_sigs = []

    all_bands, all_hjorth, all_wt       = [], [], []
    ll_list, se_list, sr_list            = [], [], []
    samp_e, perm_e, hfd_list, flat_list  = [], [], [], []
    all_std                               = []

    for ch in channels:
        sig  = np.array(signals_dict[ch], dtype=np.float64)
        sigf = bandpass_filter(sig, sr, 0.5, min(50.0, sr / 2 - 1))
        proc_sigs.append(sigf)
        all_bands.append(_band_powers(sigf, sr))
        all_hjorth.append(_hjorth(sigf))
        all_wt.append(compute_wavelet_features(sigf))
        ll_list.append(_line_length(sigf))
        se_list.append(_spectral_edge(sigf, sr))
        sr_list.append(_spike_rate(sigf, sr))
        samp_e.append(_sample_entropy(sigf))
        perm_e.append(_permutation_entropy(sigf))
        hfd_list.append(_higuchi_fd(sigf))
        flat_list.append(_spectral_flatness(sigf, sr))
        all_std.append(float(np.std(sigf)))

    sync = _inter_channel_sync(proc_sigs, sr)

    def avg(lst):    return float(np.mean(lst))
    def avgd(dicts): return {k: float(np.mean([d.get(k, 0) for d in dicts]))
                             for k in dicts[0]}

    ab  = avgd(all_bands)
    awt = avgd(all_wt) if all_wt and all_wt[0] else {}

    avg_kurt    = avg([kurtosis(s) for s in proc_sigs])
    avg_std     = avg(all_std)
    avg_amp     = avg([float(np.max(np.abs(s))) for s in proc_sigs])
    ch_std_var  = float(np.var(all_std))
    avg_ll      = avg(ll_list)
    norm_ll     = avg_ll / (avg_std * len(proc_sigs[0]) + 1e-9)
    avg_se      = avg(se_list)
    avg_act, avg_mob, avg_cpx = [avg([h[i] for h in all_hjorth]) for i in range(3)]
    avg_sr      = avg(sr_list)
    avg_se_ent  = avg(samp_e)
    avg_pe      = avg(perm_e)
    avg_hfd     = avg(hfd_list)
    avg_flat    = avg(flat_list)

    delta_p = ab['delta']; theta_p = ab['theta']
    alpha_p = ab['alpha']; beta_p  = ab['beta'];  gamma_p = ab['gamma']
    slow_r  = delta_p + theta_p
    sws     = (theta_p / (alpha_p + 1e-3)) * min(avg_kurt / 3, 3)
    wt_hi   = awt.get('cD1_energy_ratio', 0) + awt.get('cD2_energy_ratio', 0)
    wt_lo_e = awt.get('cD5_energy_ratio', 0) + awt.get('cA5_energy_ratio', 0)

    reasons, scores = [], {}

    # SEIZ
    s = 0.0
    if gamma_p > 20:
        s += 25 + (gamma_p - 20) * 1.5
        reasons.append(f"High gamma: {gamma_p:.1f}%")
    if norm_ll > 0.5:
        s += min(20 + norm_ll * 10, 28)
        reasons.append("High line-length (rhythmic burst)")
    if avg_cpx > 3.0: s += min((avg_cpx - 3) * 8, 18)
    if avg_hfd > 1.8:
        s += min((avg_hfd - 1.8) * 20, 15)
        reasons.append(f"High Higuchi FD: {avg_hfd:.2f}")
    if avg_se_ent > 1.5: s += min((avg_se_ent - 1.5) * 8, 12)
    if sync < 0.3:        s += 8
    if wt_hi > 0.35:      s += 8
    scores['SEIZ'] = float(np.clip(s, 0, 95))

    # SLOW
    s = 0.0
    if delta_p > 40:
        s += 28 + (delta_p - 40) * 1.1
        reasons.append(f"High delta: {delta_p:.1f}%")
    if avg_se < 8:
        s += 18 + (8 - avg_se) * 2.5
        reasons.append(f"Low spectral edge: {avg_se:.1f} Hz")
    if slow_r > 60:  s += (slow_r - 60) * 0.7
    if avg_cpx < 1.5: s += 10
    if avg_hfd < 1.3: s += 10
    if wt_lo_e > 0.5: s += 8
    if sync > 0.7:
        s += 8
        reasons.append(f"High inter-channel sync: {sync:.2f}")
    scores['SLOW'] = float(np.clip(s, 0, 95))

    # SPIKE
    s = 0.0
    if avg_kurt > 5:
        s += min(18 + (avg_kurt - 5) * 3.5, 38)
        reasons.append(f"Spike morphology (kurtosis={avg_kurt:.1f})")
    if avg_sr > 1:
        s += min(18 + avg_sr * 4.5, 32)
        reasons.append(f"Spike rate: {avg_sr:.1f}/s")
    if sws > 2:    s += min(sws * 5, 18)
    if avg_pe > 0.8: s += min((avg_pe - 0.8) * 15, 12)
    scores['SPIKE'] = float(np.clip(s, 0, 95))

    # ART
    s = 0.0
    if ch_std_var > 50:
        s += min(18 + ch_std_var * 0.4, 38)
        reasons.append(f"High inter-channel variance: {ch_std_var:.1f}")
    if avg_amp > 500:
        s += min(18 + (avg_amp - 500) * 0.04, 32)
        reasons.append(f"Non-physiological amplitude: {avg_amp:.0f}")
    if gamma_p > 40 and avg_kurt < 3:
        s += 18
        reasons.append("Flat high-frequency (muscle/noise)")
    if avg_flat > 0.6: s += min((avg_flat - 0.6) * 30, 12)
    scores['ART'] = float(np.clip(s, 0, 95))

    # NORM
    max_path = max(scores.values())
    s = 0.0
    if alpha_p > 20:
        s += 38 + (alpha_p - 20) * 0.7
        reasons.append(f"Alpha dominance: {alpha_p:.1f}%")
    elif alpha_p > 10:
        s += 20
    if 1.0 < avg_cpx < 3.0:   s += 14
    if avg_sr < 0.5:           s += 10
    if avg_se > 10:            s += 10
    if 0.3 < avg_pe < 0.75:   s += 8
    if 1.3 < avg_hfd < 1.8:   s += 8
    if 0.3 < sync < 0.7:      s += 6
    s *= max(0.05, 1.0 - max_path / 120.0)
    scores['NORM'] = float(np.clip(s, 5.0, 90.0))

    total      = sum(scores.values())
    probs      = {k: round(v / total * 100, 2) for k, v in scores.items()}
    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]

    if not reasons:
        reasons.append("No strong abnormality pattern detected")

    return {
        'prediction':    prediction,
        'class_name':    EEG_LABEL_NAMES.get(prediction, prediction),
        'confidence':    round(confidence, 2),
        'probabilities': probs,
        'model': 'Classic ML v3 — EEG (Spectral + Hjorth + SampEn + PermEn + HFD + Wavelet + Coherence)',
        'details': {
            'band_powers':         {k: round(v, 2) for k, v in ab.items()},
            'spectral_edge_hz':    round(avg_se, 2),
            'hjorth_complexity':   round(avg_cpx, 3),
            'higuchi_fd':          round(avg_hfd, 3),
            'sample_entropy':      round(avg_se_ent, 3),
            'permutation_entropy': round(avg_pe, 3),
            'avg_kurtosis':        round(avg_kurt, 3),
            'spike_rate_per_s':    round(avg_sr, 3),
            'spectral_flatness':   round(avg_flat, 3),
            'inter_ch_sync':       round(sync, 3),
            'slow_ratio':          round(slow_r, 2),
            'ch_std_variance':     round(ch_std_var, 3),
            'n_channels':          len(channels),
            'reasons':             reasons,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN  —  Mode A end-to-end example
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ── Mode A: train binary classifier on CPSC-2018 dataset ──────────────
    df    = load_labels(HEA_DIR)
    X, y  = build_dataset(df)
    model = train_model(X, y)
    save_model(model, feature_names=list(X.columns), save_dir='./model')

    # Binary inference on a single file
    # result = predict_mat('/path/to/recording.mat', model_dir='./model')

    # ── Mode B: multi-class ECG from raw arrays ────────────────────────────
    # signals = {'II': np.random.randn(5000)}
    # result  = classify_ecg_classic(signals, sr=500)
    # print(result)

    # ── Mode C: multi-class EEG from raw arrays ────────────────────────────
    # signals = {'Fp1': np.random.randn(5000), 'Fp2': np.random.randn(5000)}
    # result  = classify_eeg_classic(signals, sr=256)
    # print(result)