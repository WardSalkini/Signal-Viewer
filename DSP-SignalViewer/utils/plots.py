"""
Plots Module
Handles creating various plot types for signal visualization
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple


# Color palette for channels
COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Yellow-green
    '#17becf',  # Cyan
]


def get_color(index: int) -> str:
    """Get color for a channel index."""
    return COLORS[index % len(COLORS)]


def create_time_plot(signals: Dict[str, np.ndarray], 
                     sample_rate: float,
                     selected_channels: List[str],
                     start_idx: int = 0,
                     end_idx: Optional[int] = None,
                     title: str = "Signal vs Time") -> go.Figure:
    """
    Create a time-domain plot of signals.
    
    Args:
        signals: Dictionary of channel_name -> signal array
        sample_rate: Sampling rate in Hz
        selected_channels: List of channel names to plot
        start_idx: Start index for plotting
        end_idx: End index for plotting (None for all)
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    for i, channel in enumerate(selected_channels):
        if channel not in signals:
            continue
            
        signal = signals[channel]
        
        if end_idx is None:
            end_idx = len(signal)
        
        signal_slice = signal[start_idx:end_idx]
        time = np.arange(start_idx, min(end_idx, len(signal))) / sample_rate
        
        fig.add_trace(go.Scatter(
            x=time,
            y=signal_slice,
            mode='lines',
            name=channel,
            line=dict(color=get_color(i), width=1.5)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        margin=dict(l=50, r=100, t=50, b=50),
        height=400
    )
    
    return fig


def create_channel_vs_channel_plot(signals: Dict[str, np.ndarray],
                                    channel_x: str,
                                    channel_y: str,
                                    start_idx: int = 0,
                                    end_idx: Optional[int] = None,
                                    title: str = "Channel vs Channel") -> go.Figure:
    """
    Create a plot of one channel vs another.
    
    Args:
        signals: Dictionary of channel_name -> signal array
        channel_x: Channel name for X axis
        channel_y: Channel name for Y axis
        start_idx: Start index
        end_idx: End index
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if channel_x not in signals or channel_y not in signals:
        fig = go.Figure()
        fig.add_annotation(text="Please select valid channels", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    sig_x = signals[channel_x]
    sig_y = signals[channel_y]
    
    # Use minimum length
    min_len = min(len(sig_x), len(sig_y))
    if end_idx is None:
        end_idx = min_len
    end_idx = min(end_idx, min_len)
    
    x_data = sig_x[start_idx:end_idx]
    y_data = sig_y[start_idx:end_idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name=f"{channel_x} vs {channel_y}",
        line=dict(color=COLORS[0], width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=channel_x,
        yaxis_title=channel_y,
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_polar_plot(signals: Dict[str, np.ndarray],
                      sample_rate: float,
                      selected_channels: List[str],
                      visible_channels: List[str],
                      period: float = 1.0,
                      start_idx: int = 0,
                      end_idx: Optional[int] = None,
                      title: str = "Polar Plot") -> go.Figure:
    """
    Create a polar plot of signals.
    The angle represents time (one revolution = period seconds).
    
    Args:
        signals: Dictionary of channel_name -> signal array
        sample_rate: Sampling rate in Hz
        selected_channels: All selected channel names
        visible_channels: Channels that are visible (not hidden)
        period: Time period for one revolution in seconds
        start_idx: Start index
        end_idx: End index
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    for i, channel in enumerate(selected_channels):
        if channel not in signals:
            continue
            
        signal = signals[channel]
        
        if end_idx is None:
            end_idx = len(signal)
        
        signal_slice = signal[start_idx:end_idx]
        n_samples = len(signal_slice)
        
        # Time array
        time = np.arange(n_samples) / sample_rate
        
        # Convert time to angle (degrees)
        # One period = 360 degrees
        theta = (time / period) * 360 % 360
        
        # Radius is the absolute signal value (shifted to be positive)
        r = signal_slice - np.min(signal_slice) + 0.1  # Shift to positive
        
        visible = channel in visible_channels
        
        fig.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            mode='lines',
            name=channel,
            line=dict(color=get_color(i), width=1.5),
            visible=True if visible else 'legendonly'
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        height=500,
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['0°', '90°', '180°', '270°']
            )
        )
    )
    
    return fig


def create_polar_ratio_plot(signals: Dict[str, np.ndarray],
                            sample_rate: float,
                            channel1: str,
                            channel2: str,
                            period: float = 1.0,
                            start_idx: int = 0,
                            end_idx: Optional[int] = None,
                            title: str = "Polar Ratio Plot") -> go.Figure:
    """
    Create a polar plot where r = |Channel1| / |Channel2|.
    
    Args:
        signals: Dictionary of channel_name -> signal array
        sample_rate: Sampling rate in Hz
        channel1: Numerator channel name
        channel2: Denominator channel name
        period: Time period for one revolution in seconds
        start_idx: Start index
        end_idx: End index
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if channel1 not in signals or channel2 not in signals:
        fig.add_annotation(text="Please select valid channels", 
                          xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    sig1 = signals[channel1]
    sig2 = signals[channel2]
    
    # Use minimum length
    min_len = min(len(sig1), len(sig2))
    if end_idx is None:
        end_idx = min_len
    end_idx = min(end_idx, min_len)
    
    sig1_slice = sig1[start_idx:end_idx]
    sig2_slice = sig2[start_idx:end_idx]
    n_samples = len(sig1_slice)
    
    # Time array
    time = np.arange(n_samples) / sample_rate
    
    # Convert time to angle (degrees)
    theta = (time / period) * 360 % 360
    
    # Calculate ratio: r = |Ch1| / |Ch2|
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    r = np.abs(sig1_slice) / (np.abs(sig2_slice) + epsilon)
    
    # Clip extreme values for visualization
    r = np.clip(r, 0, np.percentile(r, 95) * 2)
    
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        name=f"|{channel1}| / |{channel2}|",
        line=dict(color=COLORS[0], width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        showlegend=True,
        height=500,
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 90, 180, 270],
                ticktext=['0°', '90°', '180°', '270°']
            )
        )
    )
    
    return fig
