"""
ECG Classifier using ecglib pretrained models.

Uses ResNet1D-50 pretrained on 500,000+ ECG records from ecglib.
Supports 4 pathologies with binary classification for each:
  - STACH:  Sinus Tachycardia
  - SBRAD:  Sinus Bradycardia
  - IRBBB:  Incomplete Right Bundle Branch Block
  - CRBBB:  Complete Right Bundle Branch Block

Input: 12-lead ECG at 500Hz
Reference: Avetisyan et al. (2023) - Deep Neural Networks for 12-lead ECG Classification
"""

import torch
import numpy as np
import os

# Pathologies supported by ecglib pretrained models
PATHOLOGIES = ['STACH', 'SBRAD', 'IRBBB', 'CRBBB']

PATHOLOGY_NAMES = {
    'STACH': 'Sinus Tachycardia',
    'SBRAD': 'Sinus Bradycardia',
    'IRBBB': 'Incomplete Right Bundle Branch Block',
    'CRBBB': 'Complete Right Bundle Branch Block',
}

MODEL_NAME = 'resnet1d50'
MODEL_FREQUENCY = 500  # ecglib models expect 500Hz


class ECGClassifier:
    """
    ECG Classifier using ecglib pretrained ResNet1D-50 models.
    
    Loads one pretrained model per pathology. Each model does binary
    classification (pathology present / not present).
    """
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = {}
        self.pretrained = False
        
        try:
            from ecglib.models.model_builder import create_model
            
            for pathology in PATHOLOGIES:
                try:
                    model = create_model(
                        model_name=MODEL_NAME,
                        pathology=pathology,
                        pretrained=True
                    )
                    model.eval()
                    self.models[pathology] = model
                    print(f"[ECGClassifier] Loaded {pathology} model OK")
                except Exception as e:
                    print(f"[ECGClassifier] Failed to load {pathology}: {e}")
            
            if self.models:
                self.pretrained = True
                print(f"[ECGClassifier] Loaded {len(self.models)}/{len(PATHOLOGIES)} pretrained models (ecglib {MODEL_NAME})")
            else:
                print("[ECGClassifier] ERROR: No models loaded!")
                self.pretrained = False
        except Exception as e:
            import traceback
            print(f"[ECGClassifier] Warning: Could not load ecglib models: {e}")
            traceback.print_exc()
            self.pretrained = False
    
    def preprocess(self, signals_dict, sr):
        """
        Preprocess multi-channel signal for ecglib model input.
        
        ecglib expects: 12-lead ECG at 500Hz
        Input shape: (12, n_samples)
        
        Args:
            signals_dict: dict of {channel_name: [values]} 
            sr: sample rate
            
        Returns:
            torch.Tensor of shape (1, 12, n_samples_at_500Hz)
        """
        channels = list(signals_dict.keys())
        n_channels = len(channels)
        
        # Build signal array
        signals = []
        for ch in channels:
            sig = np.array(signals_dict[ch], dtype=np.float32)
            signals.append(sig)
        signals = np.array(signals)  # (n_channels, n_samples)
        
        # Resample to 500Hz if needed
        if sr != MODEL_FREQUENCY:
            from scipy.signal import resample
            n_target = int(signals.shape[1] * MODEL_FREQUENCY / sr)
            signals = np.array([resample(s, n_target) for s in signals])
        
        # Use 10 seconds of data (5000 samples at 500Hz)
        target_len = MODEL_FREQUENCY * 10
        if signals.shape[1] >= target_len:
            signals = signals[:, :target_len]
        else:
            pad_width = target_len - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
        
        # Pad to 12 leads if fewer channels
        if n_channels < 12:
            padding = np.zeros((12 - n_channels, signals.shape[1]), dtype=np.float32)
            signals = np.vstack([signals, padding])
        elif n_channels > 12:
            signals = signals[:12]
        
        # Normalize each channel (z-score)
        for i in range(signals.shape[0]):
            std = np.std(signals[i])
            if std > 0:
                signals[i] = (signals[i] - np.mean(signals[i])) / std
        
        tensor = torch.FloatTensor(signals).unsqueeze(0)  # (1, 12, samples)
        return tensor
    
    def classify(self, signals_dict, sr):
        """
        Classify ECG signal for all 4 pathologies.
        
        Each ecglib model is an independent binary classifier that outputs
        a sigmoid score: P(pathology present). These are NOT mutually exclusive.
        
        To build a probability distribution:
        1. Get raw sigmoid score for each pathology (0-1 range)
        2. Compute P(NORMAL) = product of (1 - P(pathology_i)) for all i
        3. Normalize all scores to sum to 100%
        
        Returns:
            dict with pathology scores, top prediction, and model info
        """
        if not self.pretrained or not self.models:
            return {
                'prediction': 'UNKNOWN',
                'class_name': 'Models not loaded',
                'confidence': 0.0,
                'probabilities': {},
                'model': f'ecglib {MODEL_NAME} (NOT LOADED)',
                'pretrained': False,
                'details': {}
            }
        
        x = self.preprocess(signals_dict, sr)
        
        # Step 1: Get raw sigmoid scores from each binary classifier (0-1 range)
        raw_scores = {}  # pathology -> probability (0 to 1)
        
        with torch.no_grad():
            for pathology, model in self.models.items():
                output = model(x)
                prob = torch.sigmoid(output).numpy()[0]
                # ecglib outputs shape (batch, 1) — single sigmoid value
                score = float(prob[0])
                raw_scores[pathology] = score
        
        # Step 2: Compute NORMAL as probability that NO pathology is present
        # P(normal) = ∏(1 - P(pathology_i))
        normal_prob = 1.0
        for score in raw_scores.values():
            normal_prob *= (1.0 - score)
        # Ensure minimum floor for NORMAL
        normal_prob = max(normal_prob, 0.01)
        
        # Step 3: Build combined scores and normalize to sum to 100%
        all_scores = {'NORMAL': normal_prob}
        all_scores.update(raw_scores)
        
        total = sum(all_scores.values())
        if total > 0:
            probabilities = {k: round(v / total * 100, 2) for k, v in all_scores.items()}
        else:
            probabilities = {k: round(100.0 / len(all_scores), 2) for k in all_scores}
        
        # Step 4: Determine top prediction
        top_label = max(probabilities, key=probabilities.get)
        top_confidence = probabilities[top_label]
        
        if top_label == 'NORMAL':
            prediction = 'NORMAL'
            class_name = 'Normal Sinus Rhythm'
        else:
            prediction = top_label
            class_name = PATHOLOGY_NAMES.get(top_label, top_label)
        
        # Detected pathologies (above 10% in normalized probs)
        detected = [(k, v) for k, v in probabilities.items() if k != 'NORMAL' and v > 10]
        detected.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'prediction': prediction,
            'class_name': class_name,
            'confidence': round(top_confidence, 2),
            'probabilities': probabilities,
            'model': f'ecglib {MODEL_NAME} (pretrained)',
            'pretrained': self.pretrained,
            'details': {
                'detected_pathologies': [
                    {'code': d[0], 'name': PATHOLOGY_NAMES.get(d[0], d[0]), 'confidence': round(d[1], 2)}
                    for d in detected
                ],
                'n_models': len(self.models),
                'architecture': MODEL_NAME,
                'input_frequency': MODEL_FREQUENCY,
            }
        }
