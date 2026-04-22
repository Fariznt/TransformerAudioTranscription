"""
Role: Load and manage the MAPS dataset for training and evaluation. 
Dev note: We might support MAESTRO in the same class in the future and share code for it.
This will look like passing a dataset name as an argument to the constructor.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pretty_midi

import src.data.audio_processor as audio_processor
from src.data.midi_tokenizer import MT3Tokenizer
from src.config import config


class AudioDataset(Dataset):
    """
    PyTorch Dataset wrapper for the MAPS dataset.
    This will read audio paths and MIDI paths from the MAPS json/csv index.
    """
    def __init__(self, split: str = "train"):
        """
        Build the in-memory manifest for MAPS audio/MIDI pairs.

        Reads ``datasets/maps_index.jsonl`` (JSONL rows with ``wav`` and ``mid``
        paths relative to ``datasets/maps/``), shuffles rows once so each index maps to the same 
        sample for the lifetime of this instance, and stores ``split`` for future train/val/test
        filtering (currently unused).
        """
        self.split = split # currently doesn't make a difference

        # hardcoded paths for now, potentially move to config file later
        self.index_file = Path("datasets/maps_index.jsonl")
        self._maps_root = Path("datasets/maps")

        manifest_path = self.index_file
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MAPS manifest not found: {manifest_path}")

        # load rows from manifest
        rows: List[Dict[str, str]] = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(row)

        # Randomize once at initialization so idx -> row stays stable afterwards.
        # consequence: Dataset shuffling and initializing MAPSDataset is tightly coupled
        rng = random.Random()
        rng.shuffle(rows)
        self.rows = rows

        # Shared across all __getitem__ calls; the tokenizer's vocabulary is fixed.
        self.tokenizer = MT3Tokenizer()

    def __len__(self) -> int:
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

        wav_path = self._maps_root / wav_rel
        mid_path = self._maps_root / mid_rel
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
        wav_path, mid_path = self.get_item_paths(idx)

        # load_audio returns a 1-D float32 tensor (num_samples,), mono, resampled.
        # MAPS wavs are stereo; librosa's `mono=True` averages channels, matching MT3.
        waveform = audio_processor.load_audio(str(wav_path), config.sample_rate)
        midi = pretty_midi.PrettyMIDI(str(mid_path))

        hop_width = config.hop_width
        length = config.max_input_frames  # MT3's `max_length`; `min_length` is None -> fixed.

        # _audio_to_frames padding (MT3 `mt3/preprocessors.py`):
        #   samples = np.pad(samples, [0, frame_size - len(samples) % frame_size], mode='constant')
        pad = hop_width - waveform.shape[-1] % hop_width
        waveform = torch.nn.functional.pad(waveform, (0, pad))

        n_tokens = waveform.shape[-1] // hop_width

        # select_random_chunk with uniform_random_start=True (see
        # `single_example_select_random_chunk` in `t5/data/preprocessors.py`):
        #   start = uniform(-length + 1, n_tokens)
        #   end   = min(start + length, n_tokens)
        #   start = max(start, 0)
        # random.randint is inclusive on both ends, so use n_tokens - 1 as the upper bound.
        start_frame = random.randint(-length + 1, n_tokens - 1)
        end_frame = min(start_frame + length, n_tokens)
        start_frame = max(start_frame, 0)

        start_sample = start_frame * hop_width
        end_sample = end_frame * hop_width
        audio_segment = waveform[..., start_sample:end_sample]

        # Window bounds in seconds; tokenizer trims/zeros event times to this range.
        start_time = start_sample / config.sample_rate
        end_time = end_sample / config.sample_rate

        spectrogram = audio_processor.compute_mel_spectrogram(audio_segment, config.sample_rate)
        tokens = torch.tensor(
            self.tokenizer.encode(midi, start_time, end_time),
            dtype=torch.long,
        )

        return {"spectrogram": spectrogram, "tokens": tokens}



def create_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, validation, and test dataloaders for the MAPS dataset.
    """
    pass
