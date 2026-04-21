"""
Role: Load and manage the MAPS dataset for training and evaluation. 
Dev note: We might support MAESTRO in the same class in the future and share code for it.
This will look like passing a dataset name as an argument to the constructor.
"""
from typing import Dict, List, Tuple
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import src.data.midi_tokenizer as midi_tokenizer
import src.data.audio_processor as audio_processor

class AudioDataset(Dataset):
    """
    PyTorch Dataset wrapper for the MAPS dataset.
    This will read audio paths and MIDI paths from the MAPS json/csv index.
    """
    def __init__(self, split: str = 'train'):
        """
        ???
        """
        self.split = split # currently doesn't make a difference

        # hardcoded paths for now, potentially move to config file later
        self.index_file = Path("datasets/maps_index.jsonl")
        self._maps_root = Path("datasets/maps")

        # make sure file exists
        manifest_path = Path(index_file)
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

    def __len__(self) -> int:
        pass

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
        Returns a dictionary containing:
        - 'audio_features': Mel spectrogram tensor
        - 'labels': Tokenized MIDI events
        """
        wav_path, mid_path = self.get_item_paths(idx)
        pass

def create_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, validation, and test dataloaders for the MAPS dataset.
    """
    pass