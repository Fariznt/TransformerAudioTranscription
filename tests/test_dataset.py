"""Tests for AudioDataset.get_item_paths against the real manifest."""
from __future__ import annotations

import time
import unittest
from pathlib import Path

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
            print(f"get_item_paths latency idx={idx}: {latency_ms:.3f} ms")
            self.assertEqual(wav_path.suffix, ".wav")
            self.assertEqual(mid_path.suffix, ".mid")
            self.assertTrue(wav_path.is_file(), f"Missing wav for index {idx}: {wav_path}")
            self.assertTrue(mid_path.is_file(), f"Missing midi for index {idx}: {mid_path}")

    def test_get_item_paths_raises_index_error_for_out_of_range_index(self) -> None:
        with self.assertRaises(IndexError):
            self.dataset.get_item_paths(len(self.dataset))


if __name__ == "__main__":
    unittest.main()
