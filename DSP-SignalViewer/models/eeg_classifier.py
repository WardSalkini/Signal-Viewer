"""
EEGNet - Compact CNN for multi-channel EEG classification
Based on: Lawhern et al. (2018) - "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs"

Architecture:
  - Temporal Conv → Depthwise Conv → Separable Conv → FC
  - Designed specifically for EEG signals

Classifies EEG into 5 classes:
  0: Normal
  1: Seizure (Epileptic)
  2: Slow-wave Abnormality (e.g., Delta bursts)
  3: Spike-wave (Epileptiform discharges)
  4: Artifact/Noise anomaly

Input shape: (batch, n_channels, n_samples)
"""

import torch
import torch.nn as nn
import numpy as np
import os


CLASS_NAMES = ['Normal', 'Seizure', 'Slow-wave Abnormality', 'Spike-wave Discharge', 'Artifact Anomaly']
CLASS_SHORT = ['NORM', 'SEIZ', 'SLOW', 'SPIKE', 'ART']


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG classification.
    
    Architecture:
        1. Temporal Convolution - learns frequency filters
        2. Depthwise Convolution - learns spatial filters (across channels)
        3. Separable Convolution - learns temporal patterns
        4. Classifier - FC layer
    """
    def __init__(self, n_channels=23, n_classes=5, n_samples=256*10,
                 F1=8, F2=16, D=2, dropout_rate=0.25):
        """
        Args:
            n_channels: number of EEG channels (default 23 for CHB-MIT)
            n_classes: number of output classes
            n_samples: number of time samples (default 2560 = 10sec at 256Hz)
            F1: number of temporal filters
            F2: number of pointwise filters
            D: depth multiplier for depthwise conv
        """
        super(EEGNet, self).__init__()
        
        # Block 1: Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        
        # Block 2: Depthwise Convolution (spatial filter)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )
        
        # Block 3: Separable Convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        
        # Calculate FC input size dynamically via dry-run
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.temporal_conv(dummy)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            fc_input = x.view(1, -1).shape[1]
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        # x shape: (batch, channels, samples)
        # Add channel dim for Conv2d: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = self.classifier(x)
        return x


class EEGClassifier:
    """Wrapper class for EEG classification using EEGNet."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = None
        self.pretrained = False
        self._model_n_channels = None  # track current model's channel count
        
        # Try to load pretrained weights
        model_path = os.path.join(os.path.dirname(__file__), 'eegnet_weights.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            n_channels = checkpoint.get('n_channels', 23)
            n_samples = checkpoint.get('n_samples', 2560)
            self.model = EEGNet(n_channels=n_channels, n_classes=5, n_samples=n_samples)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.pretrained = True
            self._model_n_channels = n_channels
    
    def _init_model(self, n_channels, n_samples):
        """Initialize model with correct dimensions."""
        self.model = EEGNet(n_channels=n_channels, n_classes=5, n_samples=n_samples)
        self.model.eval()
        self._model_n_channels = n_channels
    
    def preprocess(self, signals_dict, sr):
        """
        Preprocess multi-channel EEG signal for EEGNet input.
        
        Args:
            signals_dict: dict of {channel_name: [values]}
            sr: sample rate
            
        Returns:
            torch.Tensor of shape (1, n_channels, n_samples)
        """
        channels = list(signals_dict.keys())
        n_channels = len(channels)
        
        # Get signal data
        signals = []
        for ch in channels:
            sig = np.array(signals_dict[ch], dtype=np.float32)
            signals.append(sig)
        
        signals = np.array(signals)  # (n_channels, n_samples)
        
        # Resample to 256Hz if needed
        target_sr = 256
        target_len = target_sr * 10  # 10 seconds
        
        if sr != target_sr:
            from scipy.signal import resample
            n_target = int(signals.shape[1] * target_sr / sr)
            signals = np.array([resample(s, n_target) for s in signals])
        
        # Take first 10 seconds (or pad)
        if signals.shape[1] >= target_len:
            signals = signals[:, :target_len]
        else:
            pad_width = target_len - signals.shape[1]
            signals = np.pad(signals, ((0, 0), (0, pad_width)), mode='constant')
        
        # Normalize each channel
        for i in range(signals.shape[0]):
            std = np.std(signals[i])
            if std > 0:
                signals[i] = (signals[i] - np.mean(signals[i])) / std
        
        # Re-init model when channel count changes or model doesn't exist
        if self.model is None or (not self.pretrained and self._model_n_channels != n_channels):
            self._init_model(n_channels, target_len)
        
        tensor = torch.FloatTensor(signals).unsqueeze(0)  # (1, n_channels, n_samples)
        return tensor
    
    def classify(self, signals_dict, sr):
        """
        Classify EEG signal.
        
        Note: If pretrained weights were trained on synthetic data,
        confidence is capped to reflect limited reliability.
        
        Returns:
            dict with 'prediction', 'confidence', 'probabilities', 'class_name'
        """
        x = self.preprocess(signals_dict, sr)
        
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1).numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        
        # Cap confidence when model is not trained on real clinical data
        # The pretrained weights from train_model.py use synthetic data only
        max_confidence = 95.0 if self.pretrained else 60.0
        
        raw_probabilities = {
            CLASS_SHORT[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
        
        confidence = min(float(probs[pred_idx]) * 100, max_confidence)
        
        return {
            'prediction': CLASS_SHORT[pred_idx],
            'class_name': CLASS_NAMES[pred_idx],
            'confidence': round(confidence, 2),
            'probabilities': raw_probabilities,
            'model': 'EEGNet' + (' (pretrained)' if self.pretrained else ' (synthetic weights)'),
            'pretrained': self.pretrained
        }
