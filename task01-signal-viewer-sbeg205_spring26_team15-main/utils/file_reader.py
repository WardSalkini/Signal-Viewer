"""
File Reader Module
Handles reading ECG/EEG signals from EDF and CSV files
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import io


def read_edf_file(file_buffer) -> Tuple[Dict[str, np.ndarray], float, List[str]]:
    """
    Read an EDF file and extract signals.
    
    Args:
        file_buffer: File buffer from Streamlit uploader
        
    Returns:
        Tuple of (signals_dict, sample_rate, channel_names)
        - signals_dict: Dictionary mapping channel names to signal arrays
        - sample_rate: Sampling frequency in Hz
        - channel_names: List of channel names
    """
    try:
        import pyedflib
        import tempfile
        import os
        
        # Save to temp file (pyedflib needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp:
            tmp.write(file_buffer.getvalue())
            tmp_path = tmp.name
        
        try:
            edf = pyedflib.EdfReader(tmp_path)
            
            n_channels = edf.signals_in_file
            channel_names = edf.getSignalLabels()
            sample_rate = edf.getSampleFrequency(0)  # Assume same for all channels
            
            signals_dict = {}
            for i in range(n_channels):
                signal = edf.readSignal(i)
                signals_dict[channel_names[i]] = signal
            
            edf.close()
            
            return signals_dict, sample_rate, channel_names
            
        finally:
            os.unlink(tmp_path)  # Clean up temp file
            
    except ImportError:
        raise ImportError("pyedflib is required for EDF files. Install with: pip install pyedflib")


def read_csv_file(file_buffer, time_column: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], float, List[str]]:
    """
    Read a CSV file and extract signals.
    
    Expected CSV format:
    Time,Ch1,Ch2,Ch3,...
    0.0,0.5,0.3,0.1,...
    0.001,0.6,0.4,0.2,...
    
    Args:
        file_buffer: File buffer from Streamlit uploader
        time_column: Name of the time column (auto-detected if None)
        
    Returns:
        Tuple of (signals_dict, sample_rate, channel_names)
    """
    # Read CSV
    df = pd.read_csv(file_buffer)
    
    # Find time column
    time_col = None
    time_keywords = ['time', 't', 'Time', 'TIME', 'seconds', 'sec', 'ms']
    
    if time_column and time_column in df.columns:
        time_col = time_column
    else:
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'sec', 'ms']):
                time_col = col
                break
    
    # If no time column found, create one
    if time_col is None:
        # Assume 1000 Hz sampling rate
        sample_rate = 1000.0
        time_array = np.arange(len(df)) / sample_rate
    else:
        time_array = df[time_col].values
        # Calculate sample rate from time differences
        if len(time_array) > 1:
            dt = np.median(np.diff(time_array))
            sample_rate = 1.0 / dt if dt > 0 else 1000.0
        else:
            sample_rate = 1000.0
    
    # Extract signal columns (all except time)
    signal_columns = [col for col in df.columns if col != time_col]
    
    signals_dict = {}
    for col in signal_columns:
        signals_dict[col] = df[col].values.astype(float)
    
    channel_names = signal_columns
    
    return signals_dict, sample_rate, channel_names


def read_signal_file(file_buffer, filename: str) -> Tuple[Dict[str, np.ndarray], float, List[str]]:
    """
    Auto-detect file type and read signals.
    
    Args:
        file_buffer: File buffer from Streamlit uploader
        filename: Original filename to detect type
        
    Returns:
        Tuple of (signals_dict, sample_rate, channel_names)
    """
    filename_lower = filename.lower()
    
    if filename_lower.endswith('.edf'):
        return read_edf_file(file_buffer)
    elif filename_lower.endswith('.csv'):
        return read_csv_file(file_buffer)
    else:
        # Try CSV as default
        try:
            return read_csv_file(file_buffer)
        except Exception:
            raise ValueError(f"Unsupported file format: {filename}. Please use .edf or .csv files.")


def generate_sample_data(duration: float = 10.0, sample_rate: float = 500.0, 
                         n_channels: int = 3) -> Tuple[Dict[str, np.ndarray], float, List[str]]:
    """
    Generate sample ECG-like signals for testing.
    
    Args:
        duration: Signal duration in seconds
        sample_rate: Sampling rate in Hz
        n_channels: Number of channels to generate
        
    Returns:
        Tuple of (signals_dict, sample_rate, channel_names)
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples)
    
    signals_dict = {}
    channel_names = []
    
    for i in range(n_channels):
        name = f"Channel_{i+1}"
        channel_names.append(name)
        
        # Generate ECG-like signal with different frequencies
        freq = 1.0 + i * 0.2  # Heart rate varies per channel
        
        # Simple ECG simulation (QRS-like peaks)
        signal = np.zeros(n_samples)
        
        # Add baseline
        signal += 0.1 * np.sin(2 * np.pi * 0.15 * t)  # Breathing artifact
        
        # Add QRS complexes
        beat_interval = 1.0 / freq
        beat_times = np.arange(0, duration, beat_interval)
        
        for bt in beat_times:
            idx = int(bt * sample_rate)
            if idx < n_samples - 50:
                # P wave
                signal[idx:idx+10] += 0.2 * np.sin(np.linspace(0, np.pi, 10))
                # QRS complex
                signal[idx+15:idx+20] -= 0.3
                signal[idx+20:idx+25] += 1.5
                signal[idx+25:idx+30] -= 0.5
                # T wave
                signal[idx+35:idx+50] += 0.3 * np.sin(np.linspace(0, np.pi, 15))
        
        # Add some noise
        signal += 0.05 * np.random.randn(n_samples)
        
        signals_dict[name] = signal
    
    return signals_dict, sample_rate, channel_names
