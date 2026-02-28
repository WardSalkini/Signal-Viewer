"""
Classic ML-based signal classification using statistical features.
No deep learning - uses signal processing heuristics.
"""

import numpy as np


def classify_ecg_classic(signals_dict, sr):
    """
    Classify ECG using traditional signal processing features.
    Uses R-peak detection, heart rate, and waveform statistics.
    """
    # Use first available channel
    ch = list(signals_dict.keys())[0]
    signal = np.array(signals_dict[ch], dtype=np.float64)

    # Basic statistics
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)

    # Simple R-peak detection (threshold-based)
    threshold = mean_val + 0.6 * std_val
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
            if len(peaks) == 0 or (i - peaks[-1]) > sr * 0.3:  # min 300ms between peaks
                peaks.append(i)

    n_peaks = len(peaks)
    duration = len(signal) / sr

    # Heart rate
    if n_peaks >= 2:
        rr_intervals = np.diff(peaks) / sr  # in seconds
        heart_rate = 60.0 / np.mean(rr_intervals)
        hr_variability = np.std(rr_intervals)
    else:
        heart_rate = 0
        hr_variability = 0

    # Classification logic
    reasons = []
    abnormal_score = 0

    if heart_rate > 100:
        reasons.append(f"Tachycardia: HR={heart_rate:.0f} bpm (>100)")
        abnormal_score += 30
    elif heart_rate < 60 and heart_rate > 0:
        reasons.append(f"Bradycardia: HR={heart_rate:.0f} bpm (<60)")
        abnormal_score += 25
    elif heart_rate == 0:
        reasons.append("No clear R-peaks detected")
        abnormal_score += 20

    if hr_variability > 0.2:
        reasons.append(f"High HRV: {hr_variability:.3f}s (irregular rhythm)")
        abnormal_score += 20

    if std_val > 2 * np.median(np.abs(signal)):
        reasons.append("High signal variance (possible artifact)")
        abnormal_score += 15

    normal_score = max(0, 100 - abnormal_score)
    is_normal = normal_score > 50

    return {
        'prediction': 'NORMAL' if is_normal else 'ABNORMAL',
        'class_name': 'Normal Sinus Rhythm' if is_normal else 'Abnormal ECG',
        'confidence': round(normal_score if is_normal else abnormal_score, 2),
        'probabilities': {
            'NORMAL': round(normal_score, 2),
            'ABNORMAL': round(abnormal_score, 2),
        },
        'model': 'Statistical ECG Analysis',
        'details': {
            'reasons': reasons,
            'heart_rate': heart_rate,
            'n_peaks': n_peaks,
            'hr_variability': hr_variability,
        }
    }


def classify_eeg_classic(signals_dict, sr):
    """
    Classify EEG using frequency band power analysis.
    Analyzes delta, theta, alpha, beta, gamma bands.
    """
    ch = list(signals_dict.keys())[0]
    signal = np.array(signals_dict[ch], dtype=np.float64)

    # Compute FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/sr)
    fft_vals = np.abs(np.fft.rfft(signal))
    power = fft_vals ** 2

    # Band powers
    def band_power(f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return np.sum(power[mask]) / (np.sum(power) + 1e-10)

    delta = band_power(0.5, 4)
    theta = band_power(4, 8)
    alpha = band_power(8, 13)
    beta = band_power(13, 30)
    gamma = band_power(30, 100)

    # Classification heuristics
    reasons = []
    abnormal_score = 0

    if delta > 0.5:
        reasons.append(f"Excessive delta power: {delta:.1%} (possible slow-wave abnormality)")
        abnormal_score += 30

    if gamma > 0.2:
        reasons.append(f"High gamma power: {gamma:.1%} (possible artifact or seizure)")
        abnormal_score += 25

    if beta > 0.4:
        reasons.append(f"Elevated beta: {beta:.1%} (possible anxiety/seizure)")
        abnormal_score += 20

    # Check for spike patterns (high kurtosis)
    kurtosis = np.mean(((signal - np.mean(signal)) / (np.std(signal) + 1e-10)) ** 4) - 3
    if kurtosis > 5:
        reasons.append(f"High kurtosis: {kurtosis:.1f} (spike-like activity)")
        abnormal_score += 25

    normal_score = max(0, 100 - abnormal_score)
    is_normal = normal_score > 50

    return {
        'prediction': 'NORMAL' if is_normal else 'ABNORMAL',
        'class_name': 'Normal EEG' if is_normal else 'Abnormal EEG',
        'confidence': round(normal_score if is_normal else abnormal_score, 2),
        'probabilities': {
            'NORMAL': round(normal_score, 2),
            'ABNORMAL': round(abnormal_score, 2),
        },
        'model': 'Statistical EEG Band Analysis',
        'details': {
            'reasons': reasons,
            'band_powers': {
                'delta': round(delta * 100, 1),
                'theta': round(theta * 100, 1),
                'alpha': round(alpha * 100, 1),
                'beta': round(beta * 100, 1),
                'gamma': round(gamma * 100, 1),
            }
        }
    }