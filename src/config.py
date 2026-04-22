"""
This file defines the configurable training hyperparameters and data processing
constants. It omits fixed model architecture details, focusing only on parameters 
that are meant to be tuned during the experimentation workflow.
"""
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    # Hardware / System
    device: str = "cuda"

    # --- Audio & segmentation (defaults match Hawthorne et al. 2021, Sec. 3.3 / 4) ---
    #
    # Sound in a .wav is stored as a long list of numbers (samples). sample_rate is how many
    # of those numbers represent one second of audio. Higher = finer time detail, more data.
    sample_rate: int = 16000 # kHz
    #
    # The model reads a "spectrogram" rather than raw samples. To build it, we slide a window
    # across the audio and run FFT at each position to produce one column of frequencies.
    # hop_width is how many samples we slide between positions.
    # Smaller hop = finer time resolution, but more columns = longer sequence = more memory.
    hop_width: int = 128 # how "zoomed in" the time axis of your spectrogram is
    #
    # max_input_frames caps how many spectrogram columns we feed the model in one go
    # (memory grows with the square of length). Training uses random shorter crops from
    # long pieces; this is the longest allowed crop, in frames. Max real-world length in
    # seconds ≈ max_input_frames * hop_width / sample_rate (i think).
    max_input_frames: int = 511


config = TrainingConfig()
