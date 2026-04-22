"""
Role: Audio feature extraction pipeline.
Transforms raw audio waveforms into Log-Mel Spectrograms, serving as the input
features for the Transformer Encoder.

Design notes (reconciling MT3 reference vs PyTorch):
- MT3 (Hawthorne et al. 2021) loads audio with librosa and computes the
  spectrogram with TensorFlow. We use librosa for loading (CPU-bound, matches
  the reference numerics, and gives us free stereo->mono downmixing) and
  torchaudio for the mel transform (PyTorch-native, runs on any device).
- Per MT3 `spectrograms.py` / `spectral_ops.py`, the mel config is:
      n_fft=2048, hop_length=128, n_mels=512,
      f_min=20 Hz, f_max=7600 Hz, HTK mel scale,
      magnitude spectrum (not power), log with floor 1e-5 (NOT dB).
  The last point matters: MT3 uses `log(max(x, 1e-5))`, not AmplitudeToDB.
"""
from __future__ import annotations

from functools import lru_cache

import librosa
import torch
import torchaudio


def load_audio(file_path: str, sample_rate: int) -> torch.Tensor:
    """
    Load a WAV file, resample to ``sample_rate``, and mix down to mono.

    Matches MT3's loader (`note_seq.audio_io.load_audio`, which is
    ``librosa.load(path, sr=sample_rate, mono=True)``). Stereo sources like
    MAPS and MAESTRO get their channels averaged here; downstream code never
    has to deal with a channel dimension.

    Returns:
        1-D ``torch.Tensor`` of shape ``(num_samples,)`` with dtype ``float32``.
    """
    samples, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return torch.from_numpy(samples).to(torch.float32)


# MT3 spectrogram constants (see mt3/spectrograms.py, mt3/spectral_ops.py).
_N_FFT = 2048
_HOP_LENGTH = 128
_N_MELS = 512
_F_MIN = 20.0
_F_MAX = 7600.0
_LOG_FLOOR = 1e-5


@lru_cache(maxsize=4)
def _get_mel_transform(sample_rate: int, device: torch.device) -> torchaudio.transforms.MelSpectrogram:
    """Cache one MelSpectrogram module per (sample_rate, device)."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=_N_FFT,
        hop_length=_HOP_LENGTH,
        win_length=_N_FFT,
        n_mels=_N_MELS,
        f_min=_F_MIN,
        f_max=_F_MAX,
        power=1.0,          # magnitude, not power -- matches MT3 `compute_mag`
        center=True,
        norm=None,
        mel_scale="htk",    # matches tf.signal.linear_to_mel_weight_matrix
    ).to(device)


def compute_mel_spectrogram(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Compute the MT3 log-mel spectrogram of ``waveform``.

    Args:
        waveform: ``(..., num_samples)`` float tensor.
        sample_rate: sample rate of ``waveform`` in Hz.

    Returns:
        Log-mel spectrogram with shape ``(..., T, n_mels)`` where ``T`` is the
        number of STFT frames and ``n_mels == 512``. The time axis is last-but-one
        (same convention as MT3: "inputs" have shape ``(T, n_mels)``).

    Note on T:
        With torchaudio's default ``center=True``, ``T = floor(N / hop) + 1``
        for ``N`` input samples. MT3's ``tf.signal.stft(pad_end=True)`` yields
        ``T = ceil(N / hop)``. For ``N`` already padded to a multiple of
        ``hop``, ours is one frame longer. Downstream code should not assume
        ``T == N / hop`` exactly; slice/trim if a strict match is needed.
    """
    mel_t = _get_mel_transform(sample_rate, waveform.device)
    mel = mel_t(waveform)                             # (..., n_mels, T)
    log_mel = torch.log(torch.clamp(mel, min=_LOG_FLOOR))
    return log_mel.transpose(-1, -2).contiguous()    # (..., T, n_mels)
