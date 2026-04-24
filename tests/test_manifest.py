"""Tests for scripts/build_maps_manifest.py (runs script in an isolated cwd)."""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_maps_manifest.py"


class TestBuildMapsManifest(unittest.TestCase):
    def test_unpacks_zip_and_writes_one_manifest_row_per_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            datasets = tmp / "datasets"
            datasets.mkdir(parents=True, exist_ok=True)
            zip_path = datasets / "MAPS.zip"
            with zipfile.ZipFile(zip_path, "w") as z:
                z.writestr("sub/MUS_piece.wav", b"fake-wav")
                z.writestr("sub/MUS_piece.mid", b"fake-midi")

            subprocess.run(
                [sys.executable, str(SCRIPT)],
                check=True,
                cwd=str(tmp),
            )

            manifest = tmp / "datasets" / "maps_index.jsonl"
            self.assertTrue(manifest.is_file(), "manifest should be created under cwd")
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["piece_id"], "MUS_piece")
            self.assertEqual(row["wav"], "sub/MUS_piece.wav")
            self.assertEqual(row["mid"], "sub/MUS_piece.mid")

            root = tmp / "datasets" / "maps"
            self.assertTrue((root / row["wav"]).is_file())
            self.assertTrue((root / row["mid"]).is_file())

    def test_nested_zip_under_maps_is_unpacked_before_pairing(self) -> None:
        """MAPS-style layout: outer archive only contains inner zips; wav/mid live inside."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            datasets = tmp / "datasets"
            datasets.mkdir(parents=True, exist_ok=True)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as inner:
                inner.writestr("x.wav", b"w")
                inner.writestr("x.mid", b"m")

            zip_path = datasets / "MAPS.zip"
            with zipfile.ZipFile(zip_path, "w") as outer:
                outer.writestr("subset/inner_maps.zip", buf.getvalue())

            subprocess.run(
                [sys.executable, str(SCRIPT)],
                check=True,
                cwd=str(tmp),
            )

            manifest = tmp / "datasets" / "maps_index.jsonl"
            lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["piece_id"], "x")

    def test_exits_nonzero_when_zip_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                [sys.executable, str(SCRIPT)],
                cwd=str(tmp),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
