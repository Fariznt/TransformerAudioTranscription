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
    
    # Audio Processing Parameters
    sample_rate: int = 16000
    hop_length: int = 128
    n_mels: int = 256
    
    # Tokenization Vocabulary Ranges
    max_shift_steps: int = 100
    velocity_bins: int = 127
    max_notes: int = 128
    
    # Training Hyperparameters
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100
    label_smoothing: float = 0.1

config = TrainingConfig()
