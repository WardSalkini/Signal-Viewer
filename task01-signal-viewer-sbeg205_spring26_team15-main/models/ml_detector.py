"""
Classic ML Arrhythmia Detector — Improved Version
Uses advanced signal processing (NO neural networks).

ECG Labels: NORMAL, STACH, SBRAD, IRBBB, CRBBB
EEG Labels: NORM, SEIZ, SLOW, SPIKE, ART

Improvements over v1:
  - Pan-Tompkins inspired R-peak detection (more robust)
  - Better QRS width via derivative + zero-crossing
  - Improved HRV features (SDNN, RMSSD, pNN50)
  - Rhythm regularity via RR histogram entropy
  - EEG: better per-band ratios and artifact detection
  - Score fusion uses weighted evidence model (not simple threshold)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis, entropy as scipy_entropy


# ──────────────────────────────────────────────
#  SHARED UTILITIES
# ──────────────────────────────────────────────

def bandpass_filter(sig, sr, low_hz, high_hz, order=4):
    """Zero-phase Butterworth bandpass filter."""
    nyq = sr / 2.0
    lo = np.clip(low_hz / nyq, 1e-4, 0.999)
    hi = np.clip(high_hz / nyq, 1e-4, 0.999)
    if lo >= hi:
        return sig.copy()
    b, a = scipy_signal.butter(order, [lo, hi], btype='band')
    return scipy_signal.filtfilt(b, a, sig)


def compute_signal_stats(sig):
    return {
        'mean':         float(np.mean(sig)),
        'std':          float(np.std(sig)),
        'skewness':     float(skew(sig)),
        'kurtosis':     float(kurtosis(sig)),
        'max_amplitude':float(np.max(np.abs(sig))),
    }


def compute_autocorrelation(sig):
    """Return autocorrelation periodicity score (0–1)."""
    sig = sig - np.mean(sig)
    norm = np.sum(sig ** 2)
    if norm < 1e-10:
        return 0.0
    ac = np.correlate(sig, sig, mode='full')
    ac = ac[len(ac) // 2:]
    ac /= norm
    peaks, _ = scipy_signal.find_peaks(ac[1:], height=0.1)
    return float(ac[peaks[0] + 1]) if len(peaks) > 0 else 0.0


# ──────────────────────────────────────────────
#  ECG — LABELS & NAMES
# ──────────────────────────────────────────────

ECG_PATHOLOGIES = ['NORMAL', 'STACH', 'SBRAD', 'IRBBB', 'CRBBB']
ECG_PATHOLOGY_NAMES = {
    'NORMAL': 'Normal Sinus Rhythm',
    'STACH':  'Sinus Tachycardia',
    'SBRAD':  'Sinus Bradycardia',
    'IRBBB':  'Incomplete Right Bundle Branch Block',
    'CRBBB':  'Complete Right Bundle Branch Block',
}


# ──────────────────────────────────────────────
#  ECG — R-PEAK DETECTION  (Pan-Tompkins style)
# ──────────────────────────────────────────────

def detect_r_peaks(ecg_signal, sr):
    """
    Pan-Tompkins inspired detector:
      1. Bandpass  5–15 Hz
      2. Derivative
      3. Squaring
      4. Moving-window integration
      5. Adaptive threshold peak finding
    """
    sig = np.array(ecg_signal, dtype=np.float64)

    # Step 1 — bandpass
    filtered = bandpass_filter(sig, sr, 5.0, 15.0, order=2)

    # Step 2 — derivative (5-point)
    if len(filtered) > 5:
        deriv = np.gradient(filtered)
    else:
        deriv = filtered

    # Step 3 — squaring
    squared = deriv ** 2

    # Step 4 — moving-window integration (~150 ms)
    win = max(3, int(0.15 * sr))
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode='same')

    # Step 5 — adaptive threshold
    min_dist = max(1, int(0.35 * sr))           # refractory period 350 ms
    threshold = 0.35 * np.max(integrated)
    peaks, props = scipy_signal.find_peaks(
        integrated,
        height=threshold,
        distance=min_dist
    )

    # Fallback: if too few peaks, retry with lower thresholds
    if len(peaks) < 3:
        for fallback_frac in [0.20, 0.10]:
            fb_thresh = fallback_frac * np.max(integrated)
            peaks, _ = scipy_signal.find_peaks(integrated, height=fb_thresh, distance=min_dist)
            if len(peaks) >= 3:
                break

    # Last resort: simple amplitude-based peak finding on raw signal
    if len(peaks) < 3:
        abs_sig = np.abs(sig - np.mean(sig))
        raw_thresh = np.mean(abs_sig) + 0.5 * np.std(abs_sig)
        peaks, _ = scipy_signal.find_peaks(abs_sig, height=raw_thresh, distance=min_dist)

    # Refine: find true max in ±50ms around detected peak
    refine_win = max(1, int(0.05 * sr))
    refined = []
    for p in peaks:
        s = max(0, p - refine_win)
        e = min(len(sig), p + refine_win)
        refined.append(s + int(np.argmax(np.abs(sig[s:e]))))

    return np.array(refined, dtype=int)


# ──────────────────────────────────────────────
#  ECG — HRV FEATURES
# ──────────────────────────────────────────────

def compute_hrv_features(rr_intervals):
    """
    Full HRV feature set:
      hr_mean, sdnn, rmssd, pnn50, nn50, regularity, rr_entropy
    """
    empty = {'hr_mean': 0, 'sdnn': 0, 'rmssd': 0,
             'pnn50': 0, 'nn50': 0, 'regularity': 0, 'rr_entropy': 0}
    if len(rr_intervals) < 3:
        return empty

    hr = 60.0 / rr_intervals
    rr_diff = np.diff(rr_intervals)

    sdnn   = float(np.std(rr_intervals, ddof=1))
    rmssd  = float(np.sqrt(np.mean(rr_diff ** 2)))
    nn50   = int(np.sum(np.abs(rr_diff) > 0.05))
    pnn50  = nn50 / len(rr_diff) * 100 if len(rr_diff) > 0 else 0.0

    # Coefficient of variation → regularity
    cv = sdnn / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 1.0
    regularity = float(np.clip(1 - cv * 4, 0, 1))

    # Symbolic entropy of RR histogram
    counts, _ = np.histogram(rr_intervals, bins=min(10, len(rr_intervals)))
    probs = counts / counts.sum()
    rr_entropy = float(scipy_entropy(probs + 1e-9))

    return {
        'hr_mean':   float(np.mean(hr)),
        'sdnn':      sdnn,
        'rmssd':     rmssd,
        'nn50':      nn50,
        'pnn50':     pnn50,
        'regularity':regularity,
        'rr_entropy':rr_entropy,
    }


# ──────────────────────────────────────────────
#  ECG — QRS WIDTH  (derivative zero-crossing)
# ──────────────────────────────────────────────

def estimate_qrs_width(ecg_signal, r_peaks, sr):
    """
    Estimate QRS width in ms using half-max amplitude width.
    Normal <100ms | IRBBB 100–120ms | CRBBB >120ms
    """
    if len(r_peaks) < 2:
        return 90.0

    sig = np.array(ecg_signal, dtype=np.float64)
    search_ms = 80   # look ±80ms around R peak
    half = max(1, int(search_ms / 1000 * sr))

    widths = []
    for peak in r_peaks:
        s = max(0, peak - half)
        e = min(len(sig), peak + half)
        seg = np.abs(sig[s:e])
        if len(seg) < 4:
            continue

        peak_val = np.max(seg)
        if peak_val < 1e-10:
            continue

        # Half-max threshold
        threshold = peak_val * 0.5
        above = np.where(seg > threshold)[0]
        if len(above) >= 2:
            width_samples = above[-1] - above[0]
            width_ms = width_samples / sr * 1000
            if 20 < width_ms < 200:  # sanity check
                widths.append(width_ms)

    return float(np.median(widths)) if widths else 90.0


# ──────────────────────────────────────────────
#  ECG — CLASSIFIER
# ──────────────────────────────────────────────

def classify_ecg_classic(signals_dict, sr):
    """
    Classify ECG → NORMAL | STACH | SBRAD | IRBBB | CRBBB
    Uses evidence-weighted scoring, not hard if/else thresholds.
    """
    channels = list(signals_dict.keys())
    primary_sig = np.array(signals_dict[channels[0]], dtype=np.float64)

    # ── Features ──────────────────────────────
    r_peaks    = detect_r_peaks(primary_sig, sr)
    rr_itvls   = np.diff(r_peaks) / sr if len(r_peaks) > 1 else np.array([])
    hrv        = compute_hrv_features(rr_itvls)
    qrs_width  = estimate_qrs_width(primary_sig, r_peaks, sr)
    periodicity= compute_autocorrelation(primary_sig)

    hr         = hrv['hr_mean']
    regularity = hrv['regularity']
    sdnn       = hrv['sdnn']
    rmssd      = hrv['rmssd']
    reasons    = []

    # ── Raw evidence scores (0–100) ──────────
    scores = {}

    # STACH — Sinus Tachycardia
    if hr >= 100:
        s = min(50 + (hr - 100) * 2.0, 95)
        reasons.append(f"Tachycardia HR={hr:.0f} bpm")
    elif hr >= 90:
        s = (hr - 90) * 5.0
    else:
        s = 0.0
    # Regularity bonus: STACH is usually regular
    if hr >= 100 and regularity > 0.6:
        s = min(s + 10, 95)
    scores['STACH'] = s

    # SBRAD — Sinus Bradycardia
    if 0 < hr < 60:
        s = min(50 + (60 - hr) * 2.5, 95)
        reasons.append(f"Bradycardia HR={hr:.0f} bpm")
    elif 0 < hr < 65:
        s = (65 - hr) * 8.0
    else:
        s = 0.0
    if 0 < hr < 60 and regularity > 0.5:
        s = min(s + 8, 95)
    scores['SBRAD'] = s

    # IRBBB — Incomplete RBBB  (QRS 100–120 ms)
    if 100 <= qrs_width <= 120:
        s = 30 + (qrs_width - 100) * 3.0          # 30–90
        reasons.append(f"Widened QRS {qrs_width:.0f} ms (IRBBB range)")
    elif 95 <= qrs_width < 100:
        s = (qrs_width - 95) * 6.0                # soft boundary
    else:
        s = 0.0
    scores['IRBBB'] = s

    # CRBBB — Complete RBBB  (QRS > 120 ms)
    if qrs_width > 120:
        s = min(50 + (qrs_width - 120) * 2.5, 95)
        reasons.append(f"Wide QRS {qrs_width:.0f} ms (CRBBB range)")
    elif qrs_width > 115:
        s = (qrs_width - 115) * 10.0
    else:
        s = 0.0
    scores['CRBBB'] = s

    # NORMAL — build from absence of pathology + positive rhythm features
    max_path = max(scores.values()) if scores else 0.0
    normal_s  = 0.0
    if 60 <= hr <= 100:
        normal_s += 50.0
        if regularity > 0.6:
            normal_s += 20.0
        if sdnn < 0.1:
            normal_s += 10.0
        if periodicity > 0.3:
            normal_s += 10.0
        reasons.append(f"Normal HR range ({hr:.0f} bpm), regularity={regularity:.2f}")
    else:
        normal_s = max(10.0, 40.0 - max_path * 0.3)

    # Suppress NORMAL if strong pathology evidence
    normal_s = normal_s * (1 - max_path / 120.0)
    normal_s = float(np.clip(normal_s, 5.0, 90.0))
    scores['NORMAL'] = normal_s

    # Add minimum baselines (2%) for realistic uncertainty
    for k in scores:
        scores[k] = max(scores[k], 2.0)

    # ── Normalize to 100 % ────────────────────
    total = sum(scores.values())
    if total > 0:
        probs = {k: round(v / total * 100, 2) for k, v in scores.items()}
    else:
        probs = {k: round(100.0 / len(scores), 2) for k in scores}

    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]

    detected = sorted(
        [(k, v) for k, v in probs.items() if k != 'NORMAL' and v > 5],
        key=lambda x: x[1], reverse=True
    )

    return {
        'prediction':  prediction,
        'class_name':  ECG_PATHOLOGY_NAMES.get(prediction, prediction),
        'confidence':  round(confidence, 2),
        'probabilities': probs,
        'model': 'Classic ML v2 (Pan-Tompkins + HRV + QRS)',
        'details': {
            'heart_rate':     round(hr, 1),
            'regularity':     round(regularity, 3),
            'sdnn':           round(sdnn, 4),
            'rmssd':          round(rmssd, 4),
            'qrs_width_ms':   round(qrs_width, 1),
            'periodicity':    round(periodicity, 3),
            'n_peaks':        len(r_peaks),
            'reasons':        reasons,
            'detected_pathologies': [
                {'code': d[0], 'name': ECG_PATHOLOGY_NAMES.get(d[0], d[0]),
                 'confidence': round(d[1], 2)}
                for d in detected
            ]
        }
    }


# ──────────────────────────────────────────────
#  EEG — BAND POWER + FEATURES
# ──────────────────────────────────────────────

EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50),
}

EEG_LABEL_NAMES = {
    'NORM':  'Normal',
    'SEIZ':  'Seizure',
    'SLOW':  'Slow-wave Abnormality',
    'SPIKE': 'Spike-wave Discharge',
    'ART':   'Artifact Anomaly',
}


def compute_band_powers(sig, sr):
    """Return % power in each EEG band (Welch PSD)."""
    nperseg = min(len(sig), max(64, int(sr * 2)))
    freqs, psd = scipy_signal.welch(sig, fs=sr, nperseg=nperseg)
    total = np.sum(psd) + 1e-12
    return {
        band: float(np.sum(psd[(freqs >= lo) & (freqs <= hi)]) / total * 100)
        for band, (lo, hi) in EEG_BANDS.items()
    }


def compute_line_length(sig):
    """Line length — sensitive to high-freq bursts (seizure indicator)."""
    return float(np.sum(np.abs(np.diff(sig))))


def compute_spectral_edge(sig, sr, edge=0.95):
    """Frequency below which `edge` fraction of power resides."""
    nperseg = min(len(sig), max(64, int(sr * 2)))
    freqs, psd = scipy_signal.welch(sig, fs=sr, nperseg=nperseg)
    cum = np.cumsum(psd)
    cum /= cum[-1] + 1e-12
    idx = np.searchsorted(cum, edge)
    return float(freqs[min(idx, len(freqs) - 1)])


def compute_hjorth(sig):
    """Hjorth parameters: activity, mobility, complexity."""
    activity   = float(np.var(sig))
    d1         = np.diff(sig)
    mobility   = float(np.sqrt(np.var(d1) / (activity + 1e-12)))
    d2         = np.diff(d1)
    complexity = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mobility + 1e-12))
    return activity, mobility, complexity


def detect_spike_bursts(sig, sr):
    """
    Count spike-like events: samples >4 std above mean with steep rise.
    Returns spike rate per second.
    """
    z = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    spikes, _ = scipy_signal.find_peaks(z, height=4.0, distance=max(1, int(0.02 * sr)))
    return len(spikes) / (len(sig) / sr + 1e-9)


# ──────────────────────────────────────────────
#  EEG — CLASSIFIER
# ──────────────────────────────────────────────

def classify_eeg_classic(signals_dict, sr):
    """
    Classify EEG → NORM | SEIZ | SLOW | SPIKE | ART

    Evidence model:
      SEIZ  — high gamma + high line-length + high complexity
      SLOW  — high delta + low spectral edge + low complexity
      SPIKE — high kurtosis + spike rate + spike-wave ratio
      ART   — high inter-channel variance + non-physiological amplitude
      NORM  — alpha dominance + moderate complexity + low spike rate
    """
    channels = list(signals_dict.keys())
    all_bands, all_stats, ll_list, se_list, hjorth_list, spike_rates = [], [], [], [], [], []

    for ch in channels:
        sig = np.array(signals_dict[ch], dtype=np.float64)
        all_bands.append(compute_band_powers(sig, sr))
        all_stats.append(compute_signal_stats(sig))
        ll_list.append(compute_line_length(sig))
        se_list.append(compute_spectral_edge(sig, sr))
        hjorth_list.append(compute_hjorth(sig))
        spike_rates.append(detect_spike_bursts(sig, sr))

    # Averages across channels
    avg_bands = {b: np.mean([bp[b] for bp in all_bands]) for b in EEG_BANDS}
    avg_kurt       = float(np.mean([s['kurtosis']      for s in all_stats]))
    avg_std        = float(np.mean([s['std']           for s in all_stats]))
    avg_amplitude  = float(np.mean([s['max_amplitude'] for s in all_stats]))
    ch_std_var     = float(np.var( [s['std']           for s in all_stats]))

    avg_ll         = float(np.mean(ll_list))
    norm_ll        = avg_ll / (avg_std * len(signals_dict[channels[0]]) + 1e-9)
    avg_se         = float(np.mean(se_list))

    avg_activity, avg_mobility, avg_complexity = [
        float(np.mean([h[i] for h in hjorth_list])) for i in range(3)
    ]
    avg_spike_rate = float(np.mean(spike_rates))

    # Derived ratios
    slow_ratio    = avg_bands['delta'] + avg_bands['theta']
    fast_ratio    = avg_bands['beta']  + avg_bands['gamma']
    alpha_power   = avg_bands['alpha']
    delta_power   = avg_bands['delta']
    gamma_power   = avg_bands['gamma']
    theta_power   = avg_bands['theta']

    # Spike-wave: theta/alpha band with high kurtosis
    spike_wave_score = (theta_power / (alpha_power + 1e-3)) * min(avg_kurt / 3, 3)

    reasons = []
    scores  = {}

    # ── SEIZ — Seizure ────────────────────────
    seiz_s = 0.0
    if gamma_power > 20:
        seiz_s += 30 + (gamma_power - 20) * 1.5
        reasons.append(f"High gamma {gamma_power:.1f}%")
    if norm_ll > 0.5:
        seiz_s += min(20 + norm_ll * 10, 30)
        reasons.append(f"High line-length (rhythmic burst)")
    if avg_complexity > 3.0:
        seiz_s += min((avg_complexity - 3) * 8, 20)
    if avg_amplitude > avg_std * 5:
        seiz_s += 10
    scores['SEIZ'] = float(np.clip(seiz_s, 0, 95))

    # ── SLOW — Slow-wave Abnormality ──────────
    slow_s = 0.0
    if delta_power > 40:
        slow_s += 30 + (delta_power - 40) * 1.2
        reasons.append(f"High delta {delta_power:.1f}%")
    if avg_se < 8:
        slow_s += 20 + (8 - avg_se) * 3
        reasons.append(f"Low spectral edge {avg_se:.1f} Hz")
    if slow_ratio > 60:
        slow_s += (slow_ratio - 60) * 0.8
    if avg_complexity < 1.5:
        slow_s += 10
    scores['SLOW'] = float(np.clip(slow_s, 0, 95))

    # ── SPIKE — Spike-wave Discharge ──────────
    spike_s = 0.0
    if avg_kurt > 5:
        spike_s += min(20 + (avg_kurt - 5) * 4, 40)
        reasons.append(f"Spike-like morphology (kurtosis={avg_kurt:.1f})")
    if avg_spike_rate > 1:
        spike_s += min(20 + avg_spike_rate * 5, 35)
        reasons.append(f"Spike rate {avg_spike_rate:.1f}/s")
    if spike_wave_score > 2:
        spike_s += min(spike_wave_score * 5, 20)
    scores['SPIKE'] = float(np.clip(spike_s, 0, 95))

    # ── ART — Artifact Anomaly ────────────────
    art_s = 0.0
    if ch_std_var > 50:
        art_s += min(20 + ch_std_var * 0.5, 40)
        reasons.append(f"High inter-channel variance ({ch_std_var:.1f})")
    if avg_amplitude > 500:                          # physiological limit (µV)
        art_s += min(20 + (avg_amplitude - 500) * 0.05, 35)
        reasons.append(f"Non-physiological amplitude ({avg_amplitude:.0f})")
    if gamma_power > 40 and avg_kurt < 3:            # flat high-freq → muscle/line noise
        art_s += 20
        reasons.append("Flat high-frequency (muscle/noise artifact)")
    scores['ART'] = float(np.clip(art_s, 0, 95))

    # ── NORM — Normal ─────────────────────────
    norm_s = 0.0
    max_path = max(scores.values()) if scores else 0.0
    if alpha_power > 20:
        norm_s += 40 + (alpha_power - 20) * 0.8
        reasons.append(f"Alpha dominance {alpha_power:.1f}%")
    elif alpha_power > 10:
        norm_s += 20
    if 1.0 < avg_complexity < 3.0:
        norm_s += 15
    if avg_spike_rate < 0.5:
        norm_s += 10
    if avg_se > 10:
        norm_s += 10
    norm_s *= (1 - max_path / 120.0)
    norm_s = float(np.clip(norm_s, 5.0, 90.0))
    scores['NORM'] = norm_s

    # ── Normalize ─────────────────────────────
    total = sum(scores.values())
    if total > 0:
        probs = {k: round(v / total * 100, 2) for k, v in scores.items()}
    else:
        probs = {k: round(100.0 / len(scores), 2) for k in scores}

    prediction = max(probs, key=probs.get)
    confidence = probs[prediction]

    if not reasons:
        reasons.append("No strong abnormality pattern detected")

    return {
        'prediction':  prediction,
        'class_name':  EEG_LABEL_NAMES.get(prediction, prediction),
        'confidence':  round(confidence, 2),
        'probabilities': probs,
        'model': 'Classic ML v2 (Spectral + Hjorth + Spike detection)',
        'details': {
            'band_powers':      {k: round(v, 2) for k, v in avg_bands.items()},
            'spectral_edge_hz': round(avg_se, 2),
            'hjorth_complexity':round(avg_complexity, 3),
            'avg_kurtosis':     round(avg_kurt, 3),
            'spike_rate_per_s': round(avg_spike_rate, 3),
            'slow_ratio':       round(slow_ratio, 2),
            'ch_std_variance':  round(ch_std_var, 3),
            'reasons':          reasons,
        }
    }