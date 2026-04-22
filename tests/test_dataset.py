"""Tests for AudioDataset.get_item_paths against the real manifest."""
from __future__ import annotations

import time
import unittest
from pathlib import Path

import torch
import src.data.audio_processor as audio_processor
from src.config import config
from src.data.dataset import AudioDataset

# tests AudioDataset.get_item_paths against the real manifest by initializing AudioDataset
# these tests expect the manifest to be present (i.e. build_maps_manifest.py has been run properly)
class TestAudioDatasetGetItemPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = Path("datasets/maps_index.jsonl")
        if not cls.manifest_path.is_file():
            raise unittest.SkipTest("Missing datasets/maps_index.jsonl; run manifest build first.")
        cls.dataset = AudioDataset(split="train")

    def test_get_item_paths_returns_existing_wavs_for_sample_indices(self) -> None:
        self.assertGreater(len(self.dataset), 0, "Dataset must have at least one row.")

        candidate_indices = [0, len(self.dataset) // 2, len(self.dataset) - 1]
        for idx in sorted(set(candidate_indices)):
            start = time.perf_counter()
            wav_path, mid_path = self.dataset.get_item_paths(idx)
            latency_ms = (time.perf_counter() - start) * 1000.0
            print(f"get_item_paths latency for dx={idx}: {latency_ms:.3f} ms")
            print(f"wav_path: {wav_path}")
            print(f"mid_path: {mid_path}")
            self.assertEqual(wav_path.suffix, ".wav")
            self.assertEqual(mid_path.suffix, ".mid")
            self.assertTrue(wav_path.is_file(), f"Missing wav for index {idx}: {wav_path}")
            self.assertTrue(mid_path.is_file(), f"Missing midi for index {idx}: {mid_path}")

    def test_get_item_paths_raises_index_error_for_out_of_range_index(self) -> None:
        with self.assertRaises(IndexError):
            self.dataset.get_item_paths(len(self.dataset))

    def test_load_audio_with_known_maps_wav(self) -> None:
        wav_path = Path(
            "datasets/maps/StbgTGd2/ISOL/LG/MAPS_ISOL_LG_M_S1_M73_StbgTGd2.wav"
        )
        self.assertTrue(wav_path.is_file(), f"Missing test wav file: {wav_path}")

        waveform = audio_processor.load_audio(str(wav_path), config.sample_rate)

        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(waveform.dim(), 1, "Expected mono 1-D waveform tensor.")
        self.assertGreater(waveform.numel(), 0, "Waveform should not be empty.")


if __name__ == "__main__":
    unittest.main()
