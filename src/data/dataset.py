"""
Role: Load MAPS / MAESTRO-style audio+MIDI manifests for training and evaluation.

Audio / spectrogram design (reconciling MT3 reference vs PyTorch):
- MT3 (Hawthorne et al. 2021) loads audio with librosa and computes the spectrogram with
  TensorFlow. We use librosa for loading (CPU-bound, matches the reference numerics, and
  gives us free stereo->mono downmixing) and torchaudio for the mel transform (PyTorch-
  native, runs on any device).
- Per MT3 `spectrograms.py` / `spectral_ops.py`, the mel config is: n_fft=2048, hop=128,
  n_mels=512, f_min=20 Hz, f_max=7600 Hz, HTK mel scale, magnitude spectrum (not power),
  log with floor 1e-5 (NOT dB). The last point matters: MT3 uses `log(max(x, 1e-5))`,
  not AmplitudeToDB. Values are on ``TrainingConfig`` in ``src.config`` (``hop_width``, ``mel_*``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple
import json
import random
from pathlib import Path

import librosa
import pretty_midi
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from src.data.midi_tokenizer import MT3Tokenizer
from src.config import config


class AudioDataset(Dataset):
    """
    PyTorch Dataset wrapper for the MAPS dataset.
    This will read audio paths and MIDI paths from the MAPS json/csv index.
    """

    @staticmethod
    def load_audio(file_path: str, sample_rate: int) -> torch.Tensor:
        """
        Load a WAV file, resample to ``sample_rate``, and mix down to mono.

        Matches MT3's loader (`note_seq.audio_io.wav_data_to_samples_librosa`, which is
        ``librosa.load(path, sr=sample_rate, mono=True)`` under the hood). Stereo sources
        get their channels averaged here; downstream code never has to deal with a channel
        dimension.

        Returns:
            1-D ``torch.Tensor`` of shape ``(num_samples,)`` with dtype ``float32``.
        """
        samples, _ = librosa.load(file_path, sr=sample_rate, mono=True)
        return torch.from_numpy(samples).to(torch.float32)

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_mel_transform(sample_rate: int, device: torch.device) -> torchaudio.transforms.MelSpectrogram:
        """Cache one MelSpectrogram module per (sample_rate, device)."""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=config.mel_n_fft,
            hop_length=config.hop_width,
            win_length=config.mel_n_fft,
            n_mels=config.mel_n_mels,
            f_min=config.mel_f_min,
            power=1.0,
            center=True,
            norm=None,
            mel_scale="htk",
        ).to(device)

    @staticmethod
    def compute_mel_spectrogram(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Compute the MT3 log-mel spectrogram of ``waveform``.

        Args:
            waveform: ``(..., num_samples)`` float tensor.
            sample_rate: sample rate of ``waveform`` in Hz.

        Returns:
            Log-mel spectrogram with shape ``(..., T, n_mels)`` where ``T`` is the
            number of STFT frames. The time axis is 2nd to last
            (same convention as MT3: "inputs" have shape ``(T, n_mels)``).
        """
        mel_t = AudioDataset._get_mel_transform(sample_rate, waveform.device)
        mel = mel_t(waveform)
        log_mel = torch.log(torch.clamp(mel, min=config.mel_log_floor))
        return log_mel.transpose(-1, -2).contiguous()

    def __init__(
        self,
        source: str,
        split: str,
        seed: int,
        val_fraction: float = 0.1,
        test_fraction: float = 0.0,
        steps_per_epoch: int | None = None,
    ):
        """
        Build the in-memory manifest for one source/split combination.

        Args:
            source: ``"maps"`` or ``"maestro"``; selects which manifest/root to read.
            split: ``"train"``, ``"val"``, or ``"test"``; which contiguous slice of the
                (deterministically shuffled) manifest to keep.
            seed: RNG seed for the one-shot shuffle. Two instances built with the same
                ``(source, seed)`` see identical row orderings, so the same
                ``val_fraction``/``test_fraction`` always carves out the same val/test
                rows — that's what lets validation be reproducible across runs.
            val_fraction, test_fraction: fractions of the shuffled manifest reserved
                for val and test respectively. Train gets the remainder.
            steps_per_epoch: training-only virtual epoch length. When set and
                ``split="train"``, ``__len__`` reports this value and ``__getitem__``
                ignores its ``idx`` argument, picking a random row instead. Leave ``None`` to
                iterate each manifest row once per epoch.
        """
        self.is_training = split == "train"

        if source == "maps":
            index_file, self.root = config.maps_index_path, config.maps_root
        elif source == "maestro":
            index_file, self.root = config.maestro_index_path, config.maestro_root
        else:
            raise ValueError(f"Invalid source: {source}")

        if not index_file.is_file():
            raise FileNotFoundError(f"{source} manifest not found: {index_file}")

        with index_file.open("r", encoding="utf-8") as f:
            rows: List[Dict[str, str]] = [json.loads(l) for l in f if l.strip()]

        # Deterministic one-shot shuffle: same seed -> same row order -> same slices.
        random.Random(seed).shuffle(rows)
        n = len(rows)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        n_train = n - n_val - n_test
        if n_train < 0:
            raise ValueError(f"val_fraction + test_fraction > 1 for {source} (n={n})")

        if split == "train":
            self.rows = rows[:n_train]
        elif split == "val":
            self.rows = rows[n_train : n_train + n_val]
        elif split == "test":
            self.rows = rows[n_train + n_val :]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.steps_per_epoch = steps_per_epoch

        # Shared across all __getitem__ calls; the tokenizer's vocabulary is fixed.
        self.tokenizer = MT3Tokenizer()

    def __len__(self) -> int:
        if self.is_training and self.steps_per_epoch is not None:
            return self.steps_per_epoch
        return len(self.rows)

    def get_item_paths(self, idx: int) -> Tuple[Path, Path]:
        """
        Return resolved WAV and MIDI paths for a dataset index.
        Hides manifest row/key semantics from __getitem__.
        """
        if idx < 0 or idx >= len(self.rows):
            raise IndexError(f"Index out of range: {idx}")

        row = self.rows[idx]
        try:
            wav_rel = row["wav"]
            mid_rel = row["mid"]
        except KeyError as e:
            raise KeyError(f"Manifest row is missing required key: {e}") from e

        wav_path = self.root / wav_rel
        mid_path = self.root / mid_rel
        return wav_path, mid_path

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        One training example: pick one recording pair and extract a random
        crop from it, following MT3's training pipeline byte-for-byte. We
        return a log-mel spectrogram (the model input) and the MIDI event
        tokens that fall inside the same time window (the target).

        Ports two MT3 primitives into a single __getitem__:

        1. ``_audio_to_frames`` (``mt3/preprocessors.py``): pads the waveform
           up to a multiple of ``hop_width`` so sample indices line up with
           spectrogram frames. We replicate MT3's exact formula
           ``pad = frame_size - N % frame_size``, which over-pads by one full
           frame when ``N`` already divides evenly; this isn't great but
           matches the reference.

        2. ``select_random_chunk`` (``t5/data/preprocessors.py``, invoked from
           ``mt3/tasks.py`` with ``uniform_random_start=True`` and no
           ``min_length``): pick a fixed window size ``length =
           max_input_frames``, then sample a start uniformly from
           ``[-length + 1, n_tokens)`` and clamp both ends to
           ``[0, n_tokens]``. This lets the window "hang off" either edge,
           yielding crops whose length can be anywhere from 1 to ``length``
           frames -- exactly matching the paper's "single input frame to the
           maximum input length" description. The uniform-over-starts +
           boundary-clamp trick also ensures every audio frame has equal
           probability of being included in a crop, which would not be true
           if we clamped the start range to ``[0, n_tokens - length]``.

        The only MT3 step we intentionally skip is the preceding
        ``split_tokens(max_tokens_per_segment=2000)`` in ``mt3/tasks.py``.
        That splits each recording into ~2000-frame segments *before*
        caching, which is a streaming/caching optimization (not a modelling
        choice). Consequence: our training sees proportionally fewer
        boundary-truncated crops than MT3 does, because our ``n_tokens``
        (frames per full file) is much larger than 2000. The distribution
        over ``length`` conditional on being a boundary crop is unchanged.
        """
        # Virtual-epoch mode: __len__
        # reports a synthetic step count and idx is meaningless; pick a random row.
        # i think doing this will make it easier to integrate with the work on our Colab notebook
        if self.is_training and self.steps_per_epoch is not None:
            idx = int(torch.randint(0, len(self.rows), ()).item())
        wav_path, mid_path = self.get_item_paths(idx)

        # load_audio returns a 1-D float32 tensor (num_samples,), mono, resampled.
        # MAPS wavs are stereo; librosa's `mono=True` averages channels, matching MT3.
        waveform = self.load_audio(str(wav_path), config.sample_rate)
        midi = pretty_midi.PrettyMIDI(str(mid_path))

        hop = config.hop_width
        length = config.max_input_frames  # MT3's `max_length`

        # _audio_to_frames padding (MT3 `mt3/preprocessors.py`):
        #   samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode='constant')
        num_samples = waveform.shape[-1]
        # hop — leftover portion of waveform that doesn't fit into a hop-length window;
        # padding added so the waveform length is a multiple of hop_width
        pad = hop - num_samples % hop
        # add 'pad' padding values (default 0) to the right, and 0 padding values to the left
        waveform = torch.nn.functional.pad(waveform, (0, pad))

        n_tokens = num_samples // hop  # hop segments in the padded waveform (per MT3)

        # Training: select_random_chunk with uniform_random_start=True (see
        # `single_example_select_random_chunk` in `t5/data/preprocessors.py`):
        #   start = uniform(-length + 1, n_tokens)
        #   end   = min(start + length, n_tokens)
        #   start = max(start, 0)
        #
        # Eval (val/test): take the deterministic first crop of the piece. Matches
        # the reference Colab's PianoDataset eval path and keeps val loss reproducible
        # across epochs. Full-song sliding-window inference lives in the eval script,
        # not here.
        if self.is_training:
            start_frame = int(torch.randint(-length + 1, n_tokens, ()).item())
            # pick end frame so that segment is length-long or truncated at the end of the waveform
            end_frame = min(start_frame + length, n_tokens)
            # clamp start frame to 0 so that we don't start before the beginning of the waveform
            start_frame = max(start_frame, 0)
        else: 
            # Note: perhaps taking from the start biases validation, but its probably worth it
            # because it keeps validation loss reproducible across epochs
            start_frame = 0
            end_frame = min(length, n_tokens)

        # convert frame indices to sample indices
        start_sample = start_frame * hop
        end_sample = end_frame * hop
        audio_segment = waveform[..., start_sample:end_sample]  # get audio segment from waveform

        # Window bounds in seconds; tokenizer trims/zeros event times to this range.
        # convert sample indices to seconds so we can use it to tokenize the MIDI and keep
        # the right part of the MIDI that corresponds to the audio segment
        # (inside tokenizer.encode())
        start_time = start_sample / config.sample_rate
        end_time = end_sample / config.sample_rate

        spectrogram = self.compute_mel_spectrogram(audio_segment, config.sample_rate)

        # Truncate to exactly `length` frames. 
        spectrogram = spectrogram[..., :length, :]

        # Fixed-shape padding so default DataLoader collation can torch.stack a batch.
        # MT3 deviation: MT3's seqio task uses `pack=True` (mt3/tasks.py) to pack
        # multiple short examples into one fixed-length sequence. We instead right-pad
        # each example independently -- same fixed shape, but wastes padding on
        # boundary crops. Complete fidelity to the original will create unnecessary complexity.
        
        # Spectrogram: pad time axis (-2) up to `max_input_frames` with 0.0
        T = spectrogram.shape[-2]
        if T < length:
            spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, 0, length - T))

        # Tokens: truncate to leave room for EOS, append EOS, right-pad with PAD.
        ids = self.tokenizer.encode(midi, start_time, end_time)[: config.max_target_tokens - 1]
        ids.append(config.eos_id)
        tokens = torch.full((config.max_target_tokens,), config.pad_id, dtype=torch.long)
        tokens[: len(ids)] = torch.tensor(ids, dtype=torch.long)

        # Keys match the reference Colab's PianoDataset (batch["inputs"]/["targets"]),
        # so the reference's training loop / loss functions can consume us verbatim.
        return {"inputs": spectrogram, "targets": tokens}

def create_dataloaders(
    batch_size: int,
    trainval_source: str = "maestro",
    test_source: str = "maps",
    val_fraction: float = 0.1, # also the test fraction TODO: verify fractions
    seed: int = 0,
    steps_per_epoch: int | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train, val, test) dataloaders.

    Scenarios:
      - ``trainval_source="maestro", test_source="maps"``: default MT3 setup.
      - ``trainval_source=test_source="maps"``: MAPS-only (val+test carved from MAPS).
      - ``trainval_source=test_source="maestro"``: MAESTRO-only.

    When sources differ, the test source is consumed entirely as test data
    (``val_fraction=0, test_fraction=1``). When they match, a small test slice is
    also carved from the shared pool (same seed keeps train/val/test disjoint).
    """
    # Cross-source case (e.g. train/val on MAESTRO, test on MAPS).
    # Train/val datasets carve `val_fraction` out of `trainval_source`; test set
    # consumes the entire other source (val_fraction=0, test_fraction=1).
    if trainval_source != test_source:
        train_ds = AudioDataset(
            source=trainval_source,
            split="train",
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=0.0,
            steps_per_epoch=steps_per_epoch, # virtual epoch only appropriate for train
        ) 
        val_ds = AudioDataset(
            source=trainval_source,
            split="val",
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=0.0,
        )
        test_ds = AudioDataset(
            source=test_source,
            split="test",
            seed=seed,
            val_fraction=0.0,
            test_fraction=1.0, # whole manifest is test set bc train/validation on another one        
        ) 
    # Single-source case: train/val/test all carved from the same manifest.
    # Same seed across the three calls -> identical shuffle -> disjoint slices.
    else:
        test_fraction = val_fraction # default: equal val/test slice sizes
        common = dict(
            source=trainval_source,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        train_ds = AudioDataset(split="train", steps_per_epoch=steps_per_epoch, **common)
        val_ds   = AudioDataset(split="val",  **common)
        test_ds  = AudioDataset(split="test", **common)

    # Train: shuffle so batches are i.i.d. across the manifest. (In virtual-epoch
    # mode __getitem__ ignores idx anyway, but shuffle=True is harmless.)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True, # keeps batch shape stable for training (drops the last smaller batch)
    )
    # Val/test: no shuffle (reproducible metrics) and no drop_last (keep every example).
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
