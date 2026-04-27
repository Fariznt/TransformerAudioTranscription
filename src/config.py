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
    sample_rate: int = 16000  # kHz
    #
    # max_input_frames caps how many spectrogram columns we feed the model in one go
    # (memory grows with the square of length). Training uses random shorter crops from
    # long pieces; this is the longest allowed crop, in frames. Max real-world length in
    # seconds ≈ max_input_frames * hop_width / sample_rate (i think).
    max_input_frames: int = 511

    # --- MT3 log-mel spectrogram (Hawthorne et al. 2021; mt3/spectrograms.py, spectral_ops.py) ---
    # hop_width is STFT hop in samples (128 in MT3); also used for padding/crops so frames align
    # with the mel spectrogram.
    # The model reads a "spectrogram" rather than raw samples. To build it, we slide a window
    # across the audio and run FFT at each position to produce one column of frequencies.
    hop_width: int = 128 #how many samples we slide between positions.
    # Smaller hop = finer time resolution, but more columns = longer sequence = more memory.
    mel_n_fft: int = 2048 # FFT sliding window length in samples
    mel_n_mels: int = 512 #  How many mel bands (frequency "bins" on the mel scale) you keep
    # (more mels means more detail on freq axis)
    mel_f_min: float = 20.0 # lowest frequency to include in the mel spectrogram
    mel_log_floor: float = 1e-5 # Floor on linear magnitude before log (prevents near-zero values from blowing up)

    # --- Token sequence shape (matches MT3 `targets_length=1024` from `mt3/tasks.py`). ---
    # Each __getitem__ right-pads tokens up to `max_target_tokens` with `pad_id` and
    # appends `eos_id` so default DataLoader collation can stack them into (B, T_tgt).
    max_target_tokens: int = 1024
    pad_id: int = 0
    eos_id: int = 1

    # --- MIDI / tokenizer parameters ---
    midi_max: int = 127              # max value for MIDI pitch, velocity, program, drum pitch
    drum_duration_s: float = 0.05   # fixed duration for drum hits at decode time
    min_note_duration_s: float = 0.01  # minimum note duration enforced at decode time

    # --- Dataset layout (paths relative to process cwd, project root) ---
    # These paths are what are used for creating a manifest file from dataset, and loading the dataset
    maps_index_path: Path = Path("datasets/maps_index.jsonl")
    maps_root: Path = Path("datasets/maps")
    maestro_index_path: Path = Path("datasets/maestro_index.jsonl")
    maestro_root: Path = Path("datasets/maestro")


config = TrainingConfig()
