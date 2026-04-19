#!/usr/bin/env python3
"""Prepare MAPS for training: unpack under ``datasets/maps/``, then write ``datasets/maps_index.jsonl``.

Paths are relative to **cwd** (run from the repo root).

1. If ``datasets/maps/`` is empty, unpack ``datasets/MAPS.zip`` into it.
2. While there is any ``*.zip`` under ``datasets/maps/``, unpack it into the
   **same directory as that zip** (nested layout from the distributor), then
   delete the zip so the next pass finds deeper content until ``.wav`` / ``.mid``
   appear at the MAPS leaves.
3. Walk ``datasets/maps/``, pair each ``*.wav`` with same-stem ``.mid`` in the
   same folder; write one JSON object per line to ``datasets/maps_index.jsonl``.

Training code resolves ``row["mid"]`` under ``datasets/maps/`` and passes the
path to ``pretty_midi.PrettyMIDI(...)``.

Usage: ``python scripts/build_maps_manifest.py`` (with ``datasets/MAPS.zip`` present).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

MAPS_ZIP = Path("datasets/MAPS.zip")
MAPS_DIR = Path("datasets/maps")
MANIFEST = Path("datasets/maps_index.jsonl")


def _unwrap_nested_zips(root: Path) -> None:
    """Extract every zip under root into its parent directory; remove the zip after."""
    while True:
        zips = sorted(p for p in root.rglob("*.zip") if p.is_file())
        if not zips:
            break
        for zp in zips:
            print(f"Extracting nested {zp.relative_to(root)}")
            with zipfile.ZipFile(zp) as z:
                z.extractall(zp.parent)
            zp.unlink()


def main() -> None:
    zip_path = MAPS_ZIP.resolve()
    if not zip_path.is_file():
        raise SystemExit(
            f"Place the MAPS archive at {MAPS_ZIP} (relative to cwd). Missing: {zip_path}"
        )

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    if not any(MAPS_DIR.iterdir()):
        print(f"Extracting {zip_path} -> {MAPS_DIR.resolve()}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MAPS_DIR)
    else:
        print(f"Using existing tree: {MAPS_DIR.resolve()}")

    _unwrap_nested_zips(MAPS_DIR)

    root = MAPS_DIR.resolve()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with MANIFEST.open("w", encoding="utf-8") as f:
        for wav in sorted(root.rglob("*.wav")):
            mid = wav.with_suffix(".mid")
            if not mid.is_file():
                continue
            rec = {
                "piece_id": wav.stem,
                "wav": wav.relative_to(root).as_posix(),
                "mid": mid.relative_to(root).as_posix(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} rows to {MANIFEST.resolve()}")


if __name__ == "__main__":
    main()
