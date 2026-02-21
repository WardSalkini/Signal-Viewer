"""
Train ECGNet and EEGNet on synthetic data and save pretrained weights.
Run once: python models/train_model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ecg_classifier import ECGNet
from models.eeg_classifier import EEGNet


def generate_ecg_beat(sr=100, duration=0.8):
    t = np.linspace(0, duration, int(sr * duration))
    beat = np.zeros_like(t)
    beat += 0.15 * np.exp(-((t - 0.15*duration)**2) / (2*(0.02*duration)**2))
    beat -= 0.2 * np.exp(-((t - 0.35*duration)**2) / (2*(0.008*duration)**2))
    beat += 1.2 * np.exp(-((t - 0.4*duration)**2) / (2*(0.01*duration)**2))
    beat -= 0.3 * np.exp(-((t - 0.45*duration)**2) / (2*(0.008*duration)**2))
    beat += 0.3 * np.exp(-((t - 0.65*duration)**2) / (2*(0.03*duration)**2))
    return beat


def generate_ecg_signal(label, n_channels=12, n_samples=1000, sr=100):
    signal = np.zeros((n_channels, n_samples))
    hr = 70 + np.random.randn() * 5
    beat_interval = sr / (hr / 60)
    
    for ch in range(n_channels):
        amp = 0.8 + 0.4 * np.random.rand()
        beat = generate_ecg_beat(sr)
        beat_len = len(beat)
        pos = 0
        while pos + beat_len < n_samples:
            interval = beat_interval * (0.6 + 0.8*np.random.rand()) if label == 3 else beat_interval * (0.98 + 0.04*np.random.rand())
            start_idx = int(pos)
            end_idx = min(start_idx + beat_len, n_samples)
            length = end_idx - start_idx
            segment = beat[:length].copy()
            
            if label == 1 and ch in [1,2,3,7,8]:
                segment += 0.3 * (1 + 0.5*np.random.rand())
                q_idx = int(0.3*len(segment))
                if q_idx+3 < len(segment):
                    segment[max(0,q_idx-3):q_idx+3] -= 0.4
            elif label == 2 and ch in [0,1,4,5]:
                segment -= 0.2
                t_start = int(0.55*len(segment))
                segment[t_start:] = -np.abs(segment[t_start:]) * 0.5
            elif label == 4:
                amp *= 1.5
            
            signal[ch, start_idx:end_idx] += amp * segment
            pos += interval
        signal[ch] += 0.02 * np.random.randn(n_samples)
    return signal.astype(np.float32)


def generate_eeg_signal(label, n_channels=23, n_samples=2560, sr=256):
    t = np.linspace(0, n_samples/sr, n_samples)
    signal = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        amp = 0.8 + 0.4*np.random.rand()
        if label == 0:
            freq = 10 + np.random.randn()
            signal[ch] = amp*(0.6*np.sin(2*np.pi*freq*t) + 0.2*np.sin(2*np.pi*20*t) + 0.1*np.sin(2*np.pi*2*t) + 0.05*np.random.randn(n_samples))
        elif label == 1:
            onset = np.random.randint(n_samples//4, n_samples//2)
            dur = np.random.randint(n_samples//4, n_samples//2)
            signal[ch] = 0.3*np.sin(2*np.pi*10*t) + 0.05*np.random.randn(n_samples)
            end_s = min(onset+dur, n_samples)
            signal[ch, onset:end_s] += amp*3*(np.sin(2*np.pi*35*t[onset:end_s]) + 0.5*np.sin(2*np.pi*50*t[onset:end_s]))
        elif label == 2:
            signal[ch] = amp*(1.5*np.sin(2*np.pi*1.5*t) + 0.8*np.sin(2*np.pi*2.5*t) + 0.05*np.random.randn(n_samples))
        elif label == 3:
            signal[ch] = 0.3*np.sin(2*np.pi*10*t)
            spike_interval = int(sr/3)
            for start in range(0, n_samples-20, spike_interval):
                pos = max(0, start + np.random.randint(-5,5))
                if pos+15 < n_samples:
                    signal[ch,pos:pos+5] += amp*4*np.array([0.2,0.5,1.0,0.5,0.2])
                    signal[ch,pos+5:pos+15] -= amp*1.5*np.sin(np.linspace(0,np.pi,10))
            signal[ch] += 0.05*np.random.randn(n_samples)
        elif label == 4:
            signal[ch] = 0.3*np.sin(2*np.pi*10*t) + 0.05*np.random.randn(n_samples)
            if ch < 5:
                for _ in range(np.random.randint(3,8)):
                    pos = np.random.randint(0, n_samples-50)
                    signal[ch,pos:pos+50] += amp*8*(np.random.rand(50)-0.5)
    return signal.astype(np.float32)


def train_model(model, X, y, epochs=30, lr=0.001, batch_size=32):
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, pred = out.max(1)
            total += by.size(0)
            correct += pred.eq(by).sum().item()
        acc = 100.0*correct/total
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.1f}%")
    return acc


if __name__ == '__main__':
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*50)
    print("Training ECGNet...")
    print("="*50)
    X, y = [], []
    for label in range(5):
        for _ in range(300):
            X.append(generate_ecg_signal(label))
            y.append(label)
    idx = np.random.permutation(len(X))
    X_ecg = torch.FloatTensor(np.array(X)[idx])
    y_ecg = torch.LongTensor(np.array(y)[idx])
    
    ecg_model = ECGNet(n_channels=12, n_classes=5)
    acc = train_model(ecg_model, X_ecg, y_ecg, epochs=30)
    torch.save(ecg_model.state_dict(), os.path.join(save_dir, 'ecgnet_weights.pt'))
    print(f"\n>>> ECGNet saved! Accuracy: {acc:.1f}%")
    
    print("\n" + "="*50)
    print("Training EEGNet...")
    print("="*50)
    X, y = [], []
    for label in range(5):
        for _ in range(300):
            X.append(generate_eeg_signal(label))
            y.append(label)
    idx = np.random.permutation(len(X))
    X_eeg = torch.FloatTensor(np.array(X)[idx])
    y_eeg = torch.LongTensor(np.array(y)[idx])
    
    eeg_model = EEGNet(n_channels=23, n_classes=5, n_samples=2560)
    acc = train_model(eeg_model, X_eeg, y_eeg, epochs=30)
    torch.save({'model_state_dict': eeg_model.state_dict(), 'n_channels': 23, 'n_samples': 2560}, os.path.join(save_dir, 'eegnet_weights.pt'))
    print(f"\n>>> EEGNet saved! Accuracy: {acc:.1f}%")
    print("\nDone!")
