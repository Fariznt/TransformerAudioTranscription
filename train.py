# train.py — Oscar training script for piano transcription experiments.
# Usage:
#   python train.py                        # baseline (log-mel, 6 heads)
#   python train.py --input_type stft      # Exp 1: STFT input
#   python train.py --num_heads 4          # Exp 2: 4 attention heads
#
# Set OSCAR_SCRATCH to your Oscar scratch path before running.
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_type", default="log_mel", choices=["log_mel", "stft"])
parser.add_argument("--num_heads", type=int, default=6)
args = parser.parse_args()
import os, sys, json, math, random, time, gzip, pickle, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

import pretty_midi
import mir_eval
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from transformers.optimization import Adafactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({props.total_memory/1e9:.1f} GB)")
print(f"PyTorch: {torch.__version__}   Torchaudio: {torchaudio.__version__}")

# Reproducibility (not bit-exact because cuDNN/cuBLAS nondeterminism is allowed for speed).
# these CUDA libraries optimize for speed in exchange for true determinism
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


import os
SCRATCH = Path(os.environ.get("OSCAR_SCRATCH", "./piano_transcription"))
DATA_DIR  = SCRATCH / "data"
CACHE_DIR = SCRATCH / "cache"
CKPT_DIR  = SCRATCH / "checkpoints"
for d in (DATA_DIR, CACHE_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)
print(f"Data:  {DATA_DIR}")
print(f"Cache: {CACHE_DIR}")
print(f"Ckpt:  {CKPT_DIR}")

@dataclass
class Config:
    """Hyperparameters that define the audio representation, token vocabulary, model, and training loop."""

    # --- Audio -> log-mel spectrogram (paper section 4; mt3/spectrograms.py) ---
    # The model reads a "spectrogram" rather than raw samples. To build it, we slide a window
    # across the audio and run FFT at each position to produce one column of frequencies.

    # Sound in a .wav is stored as a long list of numbers (samples). sample_rate is how many
    # of those numbers represent one second of audio. Higher = finer time detail, more data.
    # Incoming waveform audio is resampled to this many samples per second.
    sample_rate: int = 16000

    # Number of raw audio samples between adjacent spectrogram frames (i.e.
    # how many samples we slide between positions when running the FFT sliding window). With
    # sample_rate=16000, 128 samples is 0.008 s, so the model sees a new frame
    # every 8 ms. Smaller hops give finer timing but more frames to process.
    hop_width: int = 128

    # Size of the short audio window analyzed by each Fourier transform. At
    # 16 kHz, 2048 samples cover 0.128 s of sound. Windows overlap heavily because
    # fft_size is much larger than hop_width.
    fft_size: int = 2048

    # Number of frequency features kept per spectrogram frame after converting
    # raw Fourier frequencies to the mel scale, a frequency scale spaced more like
    # human pitch perception. This is the input feature width before projection
    # into the Transformer's d_model dimension.
    num_mel_bins: int = 512

    # Lowest frequency, in hertz, included in the mel filterbank. 20 Hz is near
    # the lower edge of human hearing; torchaudio's default upper edge is the
    # 'Nyquist frequency', sample_rate / 2 = 8000 Hz here.
    mel_lo_hz: float = 20.0

    # --- Event vocabulary (paper section 3.2; mt3/vocabularies.py) ---
    # Time resolution used when turning note times (real) into representative tokens for time. 
    # 100 steps per second means one shift step is 0.01 s = 10 ms. During tokenization, 
    # each note time relative to the segment start is rounded to this grid.
    steps_per_second: int = 100

    # TODO cont here
    # Largest amount of time a single shift token can represent, in seconds.
    # max_shift_steps below converts this to the vocabulary value range 0..1000.
    # Longer times can be represented by emitting multiple shift tokens whose
    # values sum to the desired absolute time step.
    max_shift_seconds: int = 10

    # Number of nonzero MIDI velocity bins used for note-on events. MIDI velocity
    # is the key-strike intensity range 1..127. The codec also includes velocity
    # value 0 as a special note-off marker, so the velocity event range is 0..127.
    num_velocity_bins: int = 127

    # Inclusive MIDI pitch range represented in the vocabulary. MIDI pitch numbers
    # name keys by semitone; middle C is 60. The full 0..127 range is kept even
    # though acoustic piano notes are normally 21..108.
    min_pitch: int = 0
    max_pitch: int = 127

    # --- Sequence lengths (paper; ismir2021.gin TASK_FEATURE_LENGTHS) ---
    # Number of spectrogram frames supplied to the encoder for one training
    # example. Combined with hop_width and sample_rate, this corresponds to
    # 512 * 128 / 16000 = 4.096 s of audio. The mel transform returns 513 frames
    # for this many samples because center=True pads the waveform; the dataset
    # truncates/pads to exactly this length.
    input_length: int = 512

    # Fixed length of the decoder target sequence in model-token IDs. Each target
    # contains encoded note/time/velocity events, then EOS, then PAD tokens. If a
    # segment produces more than target_length - 1 events, it is truncated before
    # appending EOS.
    target_length: int = 1024

    # --- Transformer (T5.1.1-small; paper Table 3 / mt3/gin/model.gin) ---
    # Width of every hidden vector inside the Transformer. The 512 mel features in
    # each frame are first projected to this size, token embeddings use this size,
    # and all encoder/decoder layer inputs and outputs have this width.
    d_model: int = 512

    # Hidden width of the feed-forward block inside each Transformer layer. This
    # notebook uses T5's GEGLU feed-forward variant: it expands from d_model to
    # d_ff, applies a gated nonlinearity, then projects back to d_model.
    d_ff: int = 1024

    # Number of attention heads in each self-attention and cross-attention module.
    # Heads let the model compute several attention patterns in parallel.
    num_heads: int = 6

    # Per-head query/key/value width. The attention inner width is
    # num_heads * d_kv = 384, which intentionally differs from d_model=512 in the
    # T5-small configuration; the attention output projection maps it back to 512.
    d_kv: int = 64

    # Number of repeated encoder layers that process the input spectrogram frames.
    num_encoder_layers: int = 8

    # Number of repeated decoder layers that autoregressively predict event tokens
    # while attending to the encoder output.
    num_decoder_layers: int = 8

    # Probability of dropping activations during training in embeddings,
    # attention, and feed-forward blocks. This regularizes the model; dropout is
    # disabled automatically by model.eval() during evaluation/inference.
    dropout: float = 0.1

    # --- Loss (mt3/gin/model.gin) ---
    # Weight on the T5 z-loss term, which mildly penalizes very large logits
    # before the softmax. It helps keep the token distribution numerically stable
    # without changing the target labels.
    z_loss: float = 1e-4

    # Amount of label smoothing passed to cross entropy. 0.0 means the target
    # token is treated as fully correct and all other tokens as incorrect; larger
    # values would distribute a little target probability to other tokens.
    label_smoothing: float = 0.0

    # --- Optimizer and schedule (mt3/gin/train.gin) ---
    # Peak and post-warmup learning rate for Adafactor. The schedule ramps up from
    # near zero during warmup_steps, then stays at this value.
    learning_rate: float = 3e-4

    # Number of optimizer steps used for the linear warmup. Step 0 uses a small
    # fraction of learning_rate; after this many steps, the schedule is constant.
    warmup_steps: int = 1000

    # --- Training budget (scaled for Colab) ---
    # Number of segments processed simultaneously in one forward/backward pass.
    # Larger batches use more GPU memory and usually make gradients less noisy.
    batch_size: int = 32

    # Number of mini-batches whose gradients are accumulated before one optimizer
    # update. The effective batch size is batch_size * grad_accum.
    grad_accum: int = 1

    # Total number of optimizer updates to run before stopping training.
    total_steps: int = 11000

    # Print loss/accuracy and update the progress display every this many
    # optimizer steps.
    log_every: int = 50

    # Save latest.pt every this many optimizer steps, and also at the final step,
    # so training can resume after an interruption.
    ckpt_every: int = 100

    # Synthetic length returned by the training Dataset. Training examples are
    # sampled randomly from cached pieces, so this only controls how many random
    # examples a DataLoader epoch contains; the outer loop still stops at
    # total_steps.
    steps_per_epoch: int = 4000

    # --- Dataset selection ---
    # Which dataset supplies the train and validation pieces. Either 'maestro' or 'maps'.
    train_valid_dataset: str = "maestro"

    # Which dataset supplies the test pieces. Either 'maestro' or 'maps'.
    test_dataset: str = "maestro"

    # Apply the [min_duration_sec, max_duration_sec] duration filter to MAESTRO pieces.
    # MAPS metadata has no duration field (see maps_range_probe.py) so duration filtering
    # is always skipped for MAPS regardless of this flag.
    filter_duration: bool = True

    # --- Dataset subset ---
    # Number of metadata rows to keep from each split after (optional) duration filtering.
    # These limits reduce download, preprocessing, and training time.
    num_train_pieces: int = 100
    num_val_pieces: int = 15
    num_test_pieces: int = 15

    # Keep only pieces whose full recording duration is in this inclusive range.
    # Very short pieces provide little variety; very long pieces cost more to
    # download/preprocess and can dominate the subset size.
    # (Only applied to MAESTRO, and only when filter_duration is True.)
    min_duration_sec: float = 120.0
    max_duration_sec: float = 360.0

    # --- Automatic mixed precision ---
    # If True on CUDA, run forward/loss computation in fp16 autocast with a
    # GradScaler to save memory and improve speed. False keeps fp32 training,
    # which is closer to the paper and avoids fp16-specific numerical behavior.
    use_amp: bool = False

    # --- Experiment flags ---
    # Which spectrogram representation to feed the encoder.
    # 'log_mel': 512-bin log-mel (baseline). 'stft': log-magnitude STFT (fft_size//2+1 = 1025 bins).
    input_type: str = "log_mel"  # "log_mel" | "stft"

    @property
    def input_dim(self) -> int:
        """Feature width fed to input_proj: 512 for log-mel, 1025 for STFT."""
        return self.num_mel_bins if self.input_type == "log_mel" else self.fft_size // 2 + 1

    @property
    def segment_samples(self) -> int:
        # Number of raw waveform samples needed for one encoder input segment:
        # input_length frames * hop_width samples/frame = 65,536 samples = 4.096 s.
        return self.input_length * self.hop_width

    @property
    def num_pitches(self) -> int:
        # Count of integer MIDI pitch values in the inclusive vocabulary range.
        return self.max_pitch - self.min_pitch + 1

    @property
    def max_shift_steps(self) -> int:
        # Largest shift token value measured in 10 ms steps: 100 steps/s * 10 s.
        return self.steps_per_second * self.max_shift_seconds


config = Config(input_type=args.input_type, num_heads=args.num_heads)
print(f"Experiment: input_type={config.input_type}, num_heads={config.num_heads}")
print(f"Segment length: {config.segment_samples/config.sample_rate:.3f} s"
      f"  ({config.segment_samples} samples, {config.input_length} mel frames)")
print(f"Effective batch: {config.batch_size * config.grad_accum}"
      f"  ({config.batch_size} × grad_accum {config.grad_accum})")


# Shared HTTP Range reader used by both MAESTRO and MAPS remote zip access.
class HttpRangeReader:
    """Seekable file-like object backed by HTTP Range requests."""

    def __init__(self, url: str):
        self.url = url
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self._size = int(head.headers["Content-Length"])
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = self._size + pos
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == -1 or self._pos + n > self._size:
            n = self._size - self._pos
        if n <= 0:
            return b""

        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"},
        )
        data = urllib.request.urlopen(req).read()
        self._pos += len(data)
        return data


import posixpath, re, zipfile
from dataclasses import asdict
from typing import Iterable

MAPS_ZENODO_RECORD_URL = "https://zenodo.org/api/records/18160555"

# These are the record's individual content files. They support HTTP 206 Range
# requests and can be opened with zipfile.ZipFile(HttpRangeReader(url)).
MAPS_ARCHIVES: dict[str, str] = {
    "AkPnBcht.zip": "https://zenodo.org/api/records/18160555/files/AkPnBcht.zip/content",
    "AkPnBsdf.zip": "https://zenodo.org/api/records/18160555/files/AkPnBsdf.zip/content",
    "AkPnCGdD.zip": "https://zenodo.org/api/records/18160555/files/AkPnCGdD.zip/content",
    "AkPnStgb.zip": "https://zenodo.org/api/records/18160555/files/AkPnStgb.zip/content",
    "ENSTDkAm.zip": "https://zenodo.org/api/records/18160555/files/ENSTDkAm.zip/content",
    "ENSTDkCl.zip": "https://zenodo.org/api/records/18160555/files/ENSTDkCl.zip/content",
    "SptkBGAm.zip": "https://zenodo.org/api/records/18160555/files/SptkBGAm.zip/content",
    "SptkBGCl.zip": "https://zenodo.org/api/records/18160555/files/SptkBGCl.zip/content",
    "StbgTGd2.zip": "https://zenodo.org/api/records/18160555/files/StbgTGd2.zip/content",
}


@dataclass(frozen=True)
class MapsPiece:
    split: str
    audio_filename: str
    midi_filename: str
    canonical_composer: str
    canonical_title: str
    source_dataset: str
    maps_config: str
    maps_category: str
    archive_key: str
    archive_url: str


# HttpRangeReader is defined once in the shared helper cell immediately above.

def open_remote_zip(archive_url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpRangeReader(archive_url))

datasets_needed = {config.train_valid_dataset, config.test_dataset}
by_split_per_dataset: Dict[str, Dict[str, list]] = {} # stores dataset name -> {train: .., validation: .., test: ...} obj

if "maestro" in datasets_needed:
    # ----- OLD MAESTRO metadata flow (kept verbatim, now under conditional) -----
    META_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"
    META_PATH = DATA_DIR / "maestro-v3.0.0.json"

    if not META_PATH.exists():
        print(f"Fetching metadata → {META_PATH}")
        urllib.request.urlretrieve(META_URL, META_PATH)
    else:
        print(f"Metadata cached at {META_PATH}")

    with open(META_PATH) as f:
        raw = json.load(f)

    # Transpose column-oriented dict → list of row dicts.
    n_rows = len(raw["split"])
    pieces = [{k: raw[k][str(i)] for k in raw} for i in range(n_rows)]

    print(f"Total MAESTRO pieces: {len(pieces)}")
    print("Splits:", {s: sum(1 for p in pieces if p["split"] == s) for s in {"train","validation","test"}})


    def in_duration_range(p):
        return config.min_duration_sec <= p["duration"] <= config.max_duration_sec


    # CHANGED: duration filter is now MAESTRO-only and gated on config.filter_duration.
    by_split_maestro = {"train": [], "validation": [], "test": []} 
    for p in pieces:
        if p["split"] not in by_split_maestro:
            continue
        if config.filter_duration and not in_duration_range(p):
            continue
        by_split_maestro[p["split"]].append(p)
    by_split_per_dataset["maestro"] = by_split_maestro

if "maps" in datasets_needed:
    def split_for_maps_config(config: str) -> str:
        # Convention for AMT stress-testing: train on synthesized pianos, validate
        # on one real Disklavier subset, and test on the other.
        if config == "ENSTDkAm":
            return "test"
        if config == "ENSTDkCl":
            return "validation"
        return "train"


    def infer_maps_fields(path: str) -> tuple[str, str, str]:
        stem = posixpath.splitext(posixpath.basename(path))[0]
        category = stem.split("-")[0] if "-" in stem else "MAPS"
        match = re.search(r"_([A-Za-z0-9]+)$", stem)
        config = match.group(1) if match else ""
        return category, config, stem


    def build_metadata_for_archive(archive_key: str, archive_url: str) -> list[MapsPiece]:
        zf = open_remote_zip(archive_url)
        names = zf.namelist()
        wavs = {
            posixpath.splitext(posixpath.basename(name))[0]: name
            for name in names
            if name.lower().endswith(".wav") and "/MAPS_MUS-" in f"/{name}"
        }
        midis = {
            posixpath.splitext(posixpath.basename(name))[0]: name
            for name in names
            if name.lower().endswith((".mid", ".midi")) and "/MAPS_MUS-" in f"/{name}"
        }

        pieces: list[MapsPiece] = []
        for stem in sorted(set(wavs) & set(midis)):
            audio_filename = wavs[stem]
            midi_filename = midis[stem]
            category, config, title = infer_maps_fields(audio_filename)
            pieces.append(
                MapsPiece(
                    split=split_for_maps_config(config),
                    audio_filename=audio_filename,
                    midi_filename=midi_filename,
                    canonical_composer="MAPS",
                    canonical_title=title,
                    source_dataset="MAPS",
                    maps_config=config,
                    maps_category=category,
                    archive_key=archive_key,
                    archive_url=archive_url,
                )
            )
        return pieces


    def build_maps_metadata() -> list[MapsPiece]:
        pieces: list[MapsPiece] = []
        for archive_key, archive_url in MAPS_ARCHIVES.items():
            archive_pieces = build_metadata_for_archive(archive_key, archive_url)
            print(f"{archive_key}: {len(archive_pieces)} MAPS_MUS pairs")
            pieces.extend(archive_pieces)
        return pieces


    def to_column_oriented_json(pieces: Iterable[MapsPiece]) -> dict[str, dict[str, object]]:
        rows = [asdict(piece) for piece in pieces]
        if not rows:
            return {}
        return {key: {str(i): row[key] for i, row in enumerate(rows)} for key in rows[0]}


    def write_metadata_json(path: Path, pieces: list[MapsPiece]) -> None:
        path.write_text(
            json.dumps(to_column_oriented_json(pieces), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


    def load_metadata_json(path: Path) -> list[dict[str, object]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [{k: raw[k][str(i)] for k in raw} for i in range(len(raw["split"]))]


    MAPS_META_PATH = DATA_DIR / "maps_zenodo_metadata.json"
    if not MAPS_META_PATH.exists():
        print(f"Building MAPS metadata → {MAPS_META_PATH}")
        _maps_pieces = build_maps_metadata()
        write_metadata_json(MAPS_META_PATH, _maps_pieces)
        print(f"\nWrote {len(_maps_pieces)} rows to {MAPS_META_PATH}")
    else:
        print(f"MAPS metadata cached at {MAPS_META_PATH}")
    # Same column-oriented -> row-dict transpose as MAESTRO uses above.
    pieces_maps = load_metadata_json(MAPS_META_PATH)

    print(f"Total MAPS pieces: {len(pieces_maps)}")
    print("Splits:", {s: sum(1 for p in pieces_maps if p["split"] == s) for s in {"train","validation","test"}})

    # SKIP duration filtering for MAPS (no `duration` field in MAPS metadata, per
    # maps_range_probe.py). MAPS pieces are bucketed into splits as-is.
    by_split_maps = {"train": [], "validation": [], "test": []}
    for p in pieces_maps:
        if p["split"] in by_split_maps:
            by_split_maps[p["split"]].append(p)
    by_split_per_dataset["maps"] = by_split_maps


rng = random.Random(SEED)
for _d in by_split_per_dataset:
    for _s in by_split_per_dataset[_d]:
        rng.shuffle(by_split_per_dataset[_d][_s])

# CHANGED: train/val come from train_valid_dataset, test comes from test_dataset.
train_subset = by_split_per_dataset[config.train_valid_dataset]["train"][:config.num_train_pieces]
val_subset   = by_split_per_dataset[config.train_valid_dataset]["validation"][:config.num_val_pieces]
test_subset  = by_split_per_dataset[config.test_dataset]["test"][:config.num_test_pieces]

print(f"\nSelected: {len(train_subset)} train ({config.train_valid_dataset}) "
      f"/ {len(val_subset)} val ({config.train_valid_dataset}) "
      f"/ {len(test_subset)} test ({config.test_dataset})")

# CHANGED: the duration aggregation/print only runs when every selected piece has
# a `duration` field (true for MAESTRO, not for MAPS). MAESTRO-only branch is the
# original code verbatim.
if all("duration" in p for p in train_subset + val_subset + test_subset):
    total_sec = sum(p["duration"] for p in train_subset + val_subset + test_subset)
    print(f"Total duration: {total_sec/60:.1f} min ({total_sec/3600:.2f} h)")
    print(f"Approx WAV download size (@ ~5 MB/min): {total_sec/60*5:.0f} MB")
    print("\nTrain pieces (composer — title [duration]):")
    for p in train_subset[:5]:
        print(f"  {p['canonical_composer'][:30]:30s}  {p['canonical_title'][:50]:50s}  {p['duration']:.0f}s")
    print("  ...")
else:
    # MAPS in selection: no duration field, so just print composer/title.
    print("\nTrain pieces (composer — title):")
    for p in train_subset[:5]:
        print(f"  {p['canonical_composer'][:30]:30s}  {p['canonical_title'][:50]:50s}")
    print("  ...")


# =====================================================================
# downloads from whichever dataset(s) the config
# selects. train_subset/val_subset are from train_valid_dataset and
# test_subset is from test_dataset (built in section 4). Each subset is
# homogeneous in source, so we dispatch per-subset by its source_dataset.
#
# Result on disk is the same shape for both datasets: the file lives at
# DATA_DIR / piece["audio_filename"] (and similarly for midi_filename),
# so sections 7, 10, 20, 21 keep loading by path with no changes.
# =====================================================================


def extract_if_missing(zf, relpath: str, local_path: Path, zip_prefix: str = "", missing_location: str = "zip"):
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = zf.read(zip_prefix + relpath)
    except KeyError:
        raise FileNotFoundError(f"{relpath} not found in {missing_location}")
    local_path.write_bytes(data)


if "maestro" in datasets_needed:
    ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"


    # HttpRangeReader is defined once in the shared helper cell immediately above section 4.
    import zipfile

    # Open the remote zip. This triggers one Range request (~400 KB) for the central directory;
    # subsequent zf.read() calls each issue one more Range request for just that file.
    print("Opening MAESTRO v3 zip...  (this reads the zip index, ~400 KB)")
    _rdr = HttpRangeReader(ZIP_URL)
    _zf  = zipfile.ZipFile(_rdr)

    # Inside the zip every path is prefixed with "maestro-v3.0.0/". The metadata paths aren't.
    ZIP_PREFIX = "maestro-v3.0.0/"


    def download_maestro_subset(pieces, label):
        print(f"[{label}] downloading {len(pieces)} pieces via Range requests...")
        for p in tqdm(pieces):
            extract_if_missing(_zf, p["audio_filename"], DATA_DIR / p["audio_filename"], ZIP_PREFIX)
            extract_if_missing(_zf, p["midi_filename"],  DATA_DIR / p["midi_filename"],  ZIP_PREFIX)


# Mirrors the MAESTRO extract_if_missing / download_subset pattern, but each
# MAPS piece names its own per-piano archive (`archive_url`) instead of a single
# global zip. Group by archive_url so each remote zip is opened only once.
if "maps" in datasets_needed:
    from collections import defaultdict

    # `zipfile` and `open_remote_zip` are already in scope from section 4's Block A.

    def download_maps_subset(pieces, label):
        by_archive = defaultdict(list)
        for p in pieces:
            by_archive[p["archive_url"]].append(p)
        total = len(pieces)
        print(f"[{label}] downloading {total} MAPS pieces from {len(by_archive)} archive(s) via Range requests...")
        for archive_url, archive_pieces in by_archive.items():
            zf = open_remote_zip(archive_url)
            for p in tqdm(archive_pieces, desc=f"  {archive_pieces[0]['archive_key']}"):
                extract_if_missing(zf, p["audio_filename"], DATA_DIR / p["audio_filename"], missing_location="MAPS archive")
                extract_if_missing(zf, p["midi_filename"], DATA_DIR / p["midi_filename"], missing_location="MAPS archive")


# ----- Unified dispatcher -----
def download_subset(pieces, label):
    if not pieces:
        return
    if pieces[0].get("source_dataset") == "MAPS":
        download_maps_subset(pieces, label)
    else:
        download_maestro_subset(pieces, label)


download_subset(train_subset, "train")
download_subset(val_subset,   "val")
download_subset(test_subset,  "test")

# Sanity check
# DATA_DIR / piece["audio_filename"] just like MAESTRO files do).
first = train_subset[0]
wav = DATA_DIR / first["audio_filename"]
mid = DATA_DIR / first["midi_filename"]
print(f"\n{wav.name}: {wav.stat().st_size/1e6:.1f} MB")
print(f"{mid.name}: {mid.stat().st_size/1e3:.1f} KB")


NUM_SPECIAL = 3
PAD_ID, EOS_ID, UNK_ID = 0, 1, 2


class EventCodec:
    """Maps between (event_type, value) tuples and integer event IDs."""

    def __init__(self, cfg: Config):
        # (name, min_value, max_value) — each range emits max - min + 1 values.
        self.ranges = [
            ("shift",    0, cfg.max_shift_steps),        # 1001 values
            ("pitch",    cfg.min_pitch, cfg.max_pitch),  # 128 values
            ("velocity", 0, cfg.num_velocity_bins),      # 128 values (bin 0 = note-off)
        ]
        self._offsets: Dict[str, int] = {}
        offset = 0
        for name, lo, hi in self.ranges:
            self._offsets[name] = offset
            offset += hi - lo + 1
        self.num_classes = offset

    def encode(self, etype: str, value: int) -> int:
        lo, hi = self._value_range(etype)
        if not (lo <= value <= hi):
            raise ValueError(f"{etype}={value} not in [{lo},{hi}]")
        return self._offsets[etype] + value - lo

    def decode(self, idx: int) -> Tuple[str, int]:
        offset = 0
        for name, lo, hi in self.ranges:
            width = hi - lo + 1
            if idx < offset + width:
                return name, lo + idx - offset
            offset += width
        raise ValueError(f"token id {idx} out of range")

    def _value_range(self, etype: str) -> Tuple[int, int]:
        for name, lo, hi in self.ranges:
            if name == etype:
                return lo, hi
        raise KeyError(etype)

    def event_id_range(self, etype: str) -> Tuple[int, int]:
        lo, hi = self._value_range(etype)
        return self.encode(etype, lo), self.encode(etype, hi)

    def is_shift_id(self, idx: int) -> bool:
        lo, hi = self.event_id_range("shift")
        return lo <= idx <= hi


def to_model_id(event_id: int) -> int:
    """Event ID (as returned by codec.encode) → model token ID (shifted by NUM_SPECIAL)."""
    return event_id + NUM_SPECIAL


def from_model_id(model_id: int) -> Optional[int]:
    """Model token ID → event ID, or None if the token is a special."""
    eid = model_id - NUM_SPECIAL
    if eid < 0:
        return None
    return eid


codec = EventCodec(config)
VOCAB_SIZE = codec.num_classes + NUM_SPECIAL
VOCAB_SIZE_PADDED = 128 * math.ceil(VOCAB_SIZE / 128)

print(f"Event classes (shift+pitch+velocity): {codec.num_classes}")
print(f"Vocab size (with specials):           {VOCAB_SIZE}")
print(f"Vocab size (padded to multiple of 128): {VOCAB_SIZE_PADDED}\n")
for name, lo, hi in codec.ranges:
    first_id = codec.encode(name, lo)
    last_id  = codec.encode(name, hi)
    print(f"  {name:8s}  event IDs [{first_id:4d} .. {last_id:4d}]  ({hi-lo+1} values)")


# several MAPS MIDI files exceed pretty_midi's
# conservative default MAX_TICK=10_000_000 even though they parse to normal note
# timelines. Raise it before any MAPS MIDI is loaded by load_notes() below.
# (Setting it for MAESTRO too is harmless: MAESTRO MIDIs are well under the cap.)
if "maps" in datasets_needed:
    pretty_midi.pretty_midi.MAX_TICK = 100_000_000


def apply_sustain_pedal(midi: pretty_midi.PrettyMIDI, threshold: int = 64) -> None:
    """In-place: extend note.end to the next sustain-pedal-off event when the pedal is held.

    Port of the Magenta `apply_sustain_control_changes` logic used for MAESTRO / Onsets-and-Frames.
    """
    for inst in midi.instruments:
        events = [(cc.time, cc.value >= threshold)
                  for cc in inst.control_changes if cc.number == 64]
        events.sort()
        if not events:
            continue

        # Precompute list of off-event times for fast lookup.
        off_times = [t for t, down in events if not down]

        for note in inst.notes:
            # Find pedal state at note.end: last event whose time <= note.end.
            # Binary search would be faster but this is fine for MAESTRO's scale.
            last_down = False
            for t, down in events:
                if t <= note.end:
                    last_down = down
                else:
                    break
            if not last_down:
                continue
            # Pedal is held through note.end — extend to the next pedal-off after note.end.
            for t in off_times:
                if t > note.end:
                    note.end = t
                    break


def load_notes(midi_path: Path) -> List[Tuple[float, float, int, int]]:
    """Return sorted list of (start_sec, end_sec, pitch, velocity) for all non-drum notes."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    apply_sustain_pedal(midi)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            if n.end <= n.start:
                continue
            notes.append((float(n.start), float(n.end), int(n.pitch), int(n.velocity)))
    notes.sort()
    return notes


# Sanity check.
_first_mid = DATA_DIR / train_subset[0]["midi_filename"]
_notes = load_notes(_first_mid)
print(f"{_first_mid.name}")
print(f"  {len(_notes)} notes, final offset {_notes[-1][1]:.1f}s")
print("  first 5:")
for n in _notes[:5]:
    print(f"    start={n[0]:6.3f}  end={n[1]:6.3f}  pitch={n[2]:3d}  vel={n[3]:3d}")


def velocity_to_bin(vel: int, num_bins: int = config.num_velocity_bins) -> int:
    """Map 0..127 → 0..num_bins (0 reserved for note-off). Matches mt3/vocabularies.velocity_to_bin."""
    if vel == 0:
        return 0
    return math.ceil(num_bins * vel / 127)


def bin_to_velocity(b: int, num_bins: int = config.num_velocity_bins) -> int:
    if b == 0:
        return 0
    return int(127 * b / num_bins)


def tokenize_segment(notes: List[Tuple[float, float, int, int]],
                     start_sec: float, end_sec: float,
                     codec: EventCodec, cfg: Config) -> List[int]:
    """Return a list of event IDs (NOT model IDs) for events in [start_sec, end_sec)."""
    events = []  # (rel_time_sec, kind, pitch, velocity)
    for n_start, n_end, pitch, vel in notes:
        if start_sec <= n_start < end_sec:
            events.append((n_start - start_sec, "on", pitch, vel))
        if start_sec <= n_end < end_sec and n_start < n_end:
            events.append((n_end - start_sec, "off", pitch, 0))

    # Sort: time asc, offsets before onsets at same time, then pitch asc (mt3 order).
    events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1, e[2]))

    tokens: List[int] = []
    current_step = 0
    current_vel_bin = -1   # sentinel: force emission on first note

    for rel_sec, kind, pitch, vel in events:
        step = int(round(rel_sec * cfg.steps_per_second))
        # For our 4 s segments step <= 410 << max_shift_steps, so the chain is
        # always length 1; the while loop handles the general case.
        if step > current_step:
            remaining = step       # emit chain whose SUM = step (absolute from segment start)
            while remaining > 0:
                s = min(cfg.max_shift_steps, remaining)
                tokens.append(codec.encode("shift", s))
                remaining -= s
            current_step = step

        vbin = 0 if kind == "off" else velocity_to_bin(vel, cfg.num_velocity_bins)
        if vbin != current_vel_bin:
            tokens.append(codec.encode("velocity", vbin))
            current_vel_bin = vbin

        tokens.append(codec.encode("pitch", pitch))

    return tokens


# Sanity check on 4 s of our first piece.
_seg = tokenize_segment(_notes, 10.0, 14.0, codec, config)
print(f"Segment 10.0–14.0 s → {len(_seg)} event tokens")
for t in _seg[:18]:
    n, v = codec.decode(t)
    print(f"  id={t:4d}  {n}={v}")


if config.input_type == "log_mel":
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.fft_size,
        hop_length=config.hop_width,
        n_mels=config.num_mel_bins,
        f_min=config.mel_lo_hz,
        power=1.0,          # magnitude, not power
        mel_scale="htk",
        norm=None,
        center=True,
    )

_SPEC_EPS = 1e-6


def compute_spectrogram(samples: torch.Tensor) -> torch.Tensor:
    """samples: 1-D float tensor → [num_frames, input_dim] log spectrogram."""
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    if config.input_type == "log_mel":
        mel = mel_transform(samples).squeeze(0)    # [n_mels, n_frames]
        return torch.log(mel + _SPEC_EPS).T         # [n_frames, n_mels]
    else:  # stft
        stft = torch.stft(
            samples.squeeze(0), n_fft=config.fft_size, hop_length=config.hop_width,
            window=torch.hann_window(config.fft_size), return_complex=True,
        )  # [fft_size//2+1, n_frames]
        return torch.log(stft.abs() + _SPEC_EPS).T  # [n_frames, fft_size//2+1]


# Sanity: feed in a random segment and check the shape.
_m = compute_spectrogram(torch.randn(config.segment_samples))
print(f"spectrogram shape for {config.segment_samples}-sample segment: {tuple(_m.shape)}")
print(f"(will truncate/pad to [{config.input_length}, {config.input_dim}] in the dataset)")


def load_audio_16k(wav_path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(wav_path))
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)                     # stereo → mono
    if sr != config.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, config.sample_rate)
    return wav.squeeze(0).contiguous()                      # [T]


def _cache_path_for(piece: Dict[str, Any]) -> Path:
    return CACHE_DIR / (Path(piece["audio_filename"]).stem + f"_{config.input_type}.pkl.gz")


def preprocess_piece(piece: Dict[str, Any]) -> Dict[str, Any]:
    cp = _cache_path_for(piece)
    if cp.exists():
        with gzip.open(cp, "rb") as f:
            data = pickle.load(f)
        data["piece"] = piece
        return data

    wav_path = DATA_DIR / piece["audio_filename"]
    mid_path = DATA_DIR / piece["midi_filename"]

    audio = load_audio_16k(wav_path)
    notes = load_notes(mid_path)
    # Compute mel on CPU to avoid OOM on the GPU for long pieces.
    log_mel = compute_spectrogram(audio).cpu().half()            # fp16 to halve cache size

    data = {
        "mel":         log_mel,
        "notes":       notes,
        "duration":    float(audio.numel() / config.sample_rate),
        "num_frames":  int(log_mel.size(0)),
    }
    cp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    data["piece"] = piece
    return data


def preprocess_subset(pieces, label):
    print(f"[{label}] preprocessing {len(pieces)} pieces...")
    return [preprocess_piece(p) for p in tqdm(pieces)]


train_cache = preprocess_subset(train_subset, "train")
val_cache   = preprocess_subset(val_subset,   "val")
test_cache  = preprocess_subset(test_subset,  "test")

print(f"\nTrain mel frames: {sum(d['num_frames'] for d in train_cache):,}")
print(f"Val   mel frames: {sum(d['num_frames'] for d in val_cache):,}")
print(f"Test  mel frames: {sum(d['num_frames'] for d in test_cache):,}")


class PianoDataset(Dataset):
    def __init__(self, cached_pieces: List[Dict[str, Any]], codec: EventCodec,
                 cfg: Config, is_training: bool):
        self.pieces = cached_pieces
        self.codec = codec
        self.cfg = cfg
        self.is_training = is_training
        # A per-dataset RNG so workers (if any) can seed independently.
        self.rng = random.Random(SEED + (1 if is_training else 0))

    def __len__(self):
        if self.is_training:
            return self.cfg.steps_per_epoch
        return len(self.pieces)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        if self.is_training:
            piece = self.rng.choice(self.pieces)
            max_start = max(0, piece["num_frames"] - cfg.input_length)
            start_frame = self.rng.randint(0, max_start)
        else:
            piece = self.pieces[idx]
            start_frame = 0
        end_frame = start_frame + cfg.input_length

        # Slice the cached full-song mel (stored fp16 on CPU).
        mel = piece["mel"][start_frame:end_frame].float()
        if mel.size(0) < cfg.input_length:
            mel = F.pad(mel, (0, 0, 0, cfg.input_length - mel.size(0)))

        start_sec = start_frame * cfg.hop_width / cfg.sample_rate
        end_sec   = end_frame   * cfg.hop_width / cfg.sample_rate

        event_ids = tokenize_segment(piece["notes"], start_sec, end_sec, self.codec, cfg)
        model_ids = [to_model_id(e) for e in event_ids][: cfg.target_length - 1]
        model_ids.append(EOS_ID)

        target = torch.full((cfg.target_length,), PAD_ID, dtype=torch.long)
        target[: len(model_ids)] = torch.tensor(model_ids, dtype=torch.long)

        return {
            "inputs":  mel.contiguous(),   # [input_length, n_mels]
            "targets": target,             # [target_length]
        }


train_ds = PianoDataset(train_cache, codec, config, is_training=True)
val_ds   = PianoDataset(val_cache,   codec, config, is_training=False)

_s = train_ds[0]
print(f"inputs  {_s['inputs'].shape}   dtype {_s['inputs'].dtype}")
print(f"targets {_s['targets'].shape}  dtype {_s['targets'].dtype}")
print(f"non-pad target tokens: {(_s['targets'] != PAD_ID).sum().item()} / {config.target_length}")
print(f"first 15 target tokens: {_s['targets'][:15].tolist()}")


class T5LayerNorm(nn.Module):
    """RMSNorm: no mean centering, learned per-channel scale, no bias."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (self.weight * x32).to(in_dtype)


class FixedSinusoidalPositionalEmbedding(nn.Module):
    """Matches mt3.layers.sinusoidal() exactly: [sin | cos] halves, not interleaved."""
    def __init__(self, d_model: int, max_len: int = 4096,
                 min_scale: float = 1.0, max_scale: float = 10000.0):
        super().__init__()
        half = d_model // 2
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale_factor = -math.log(max_scale / min_scale) / (half - 1)
        div_term = min_scale * torch.exp(torch.arange(half, dtype=torch.float32) * scale_factor)
        pe[:, :half] = torch.sin(position * div_term)
        pe[:, half:2 * half] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        return x + self.pe[: x.size(1)].unsqueeze(0)


class GEGLU_MLP(nn.Module):
    """T5.1.1 feed-forward with gated-GELU. No biases; matches mt3.layers.MlpBlock."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)   # GELU branch
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)   # linear gate branch
        self.wo   = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.wi_0(x)) * self.wi_1(x)
        x = self.dropout(x)
        return self.wo(x)


class MultiHeadAttention(nn.Module):
    """T5-style multi-head dot-product attention."""
    def __init__(self, d_model: int, num_heads: int, d_kv: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_kv = d_kv
        inner = num_heads * d_kv
        self.q_proj   = nn.Linear(d_model, inner, bias=False)
        self.k_proj   = nn.Linear(d_model, inner, bias=False)
        self.v_proj   = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, H*D] → [B, H, L, D]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_kv).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, H, L, D] → [B, L, H*D]
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                is_causal: bool = False) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))          # [B,H,Lq,D]
        k = self._split_heads(self.k_proj(key_value))      # [B,H,Lk,D]
        v = self._split_heads(self.v_proj(key_value))      # [B,H,Lk,D]

        # T5: raw dot product, no sqrt(d_kv) scaling.
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k)

        if is_causal:
            Lq, Lk = q.size(2), k.size(2)
            mask = torch.ones(Lq, Lk, device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax in fp32 to avoid fp16 over/underflow.
        attn = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.dropout(attn)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        return self.out_proj(self._merge_heads(out))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.pre_attn_norm = T5LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.d_kv, cfg.dropout)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.pre_mlp_norm = T5LayerNorm(cfg.d_model)
        self.mlp = GEGLU_MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.mlp_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_attn_norm(x)
        x = x + self.attn_drop(self.attn(h, h, is_causal=False))
        h = self.pre_mlp_norm(x)
        x = x + self.mlp_drop(self.mlp(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.pre_self_norm = T5LayerNorm(cfg.d_model)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.d_kv, cfg.dropout)
        self.self_drop = nn.Dropout(cfg.dropout)
        self.pre_cross_norm = T5LayerNorm(cfg.d_model)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.d_kv, cfg.dropout)
        self.cross_drop = nn.Dropout(cfg.dropout)
        self.pre_mlp_norm = T5LayerNorm(cfg.d_model)
        self.mlp = GEGLU_MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.mlp_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        h = self.pre_self_norm(x)
        x = x + self.self_drop(self.self_attn(h, h, is_causal=True))
        h = self.pre_cross_norm(x)
        x = x + self.cross_drop(self.cross_attn(h, encoded, is_causal=False))
        h = self.pre_mlp_norm(x)
        x = x + self.mlp_drop(self.mlp(h))
        return x


class PianoTranscriptionTransformer(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size

        # --- Encoder ---
        self.input_proj  = nn.Linear(cfg.input_dim, cfg.d_model, bias=False)
        self.enc_pos     = FixedSinusoidalPositionalEmbedding(cfg.d_model, max_len=cfg.input_length + 8)
        self.enc_in_drop = nn.Dropout(cfg.dropout)
        self.enc_layers  = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.enc_norm    = T5LayerNorm(cfg.d_model)
        self.enc_out_drop = nn.Dropout(cfg.dropout)

        # --- Decoder ---
        self.tok_emb     = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0)    # matches mt3 embedding_init
        with torch.no_grad():
            self.tok_emb.weight[PAD_ID].zero_()
        self.dec_pos     = FixedSinusoidalPositionalEmbedding(cfg.d_model, max_len=cfg.target_length + 8)
        self.dec_in_drop = nn.Dropout(cfg.dropout)
        self.dec_layers  = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.dec_norm    = T5LayerNorm(cfg.d_model)
        self.dec_out_drop = nn.Dropout(cfg.dropout)

        self.logits_proj = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = self.enc_pos(x)
        x = self.enc_in_drop(x)
        for layer in self.enc_layers:
            x = layer(x)
        x = self.enc_norm(x)
        x = self.enc_out_drop(x)
        return x

    def decode(self, decoder_input_tokens: torch.Tensor, encoded: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(decoder_input_tokens)
        x = self.dec_pos(x)
        x = self.dec_in_drop(x)
        for layer in self.dec_layers:
            x = layer(x, encoded)
        x = self.dec_norm(x)
        x = self.dec_out_drop(x)
        return self.logits_proj(x)

    def forward(self, inputs: torch.Tensor, decoder_input_tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(decoder_input_tokens, self.encode(inputs))


model = PianoTranscriptionTransformer(config, vocab_size=VOCAB_SIZE_PADDED).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}  ({n_params/1e6:.1f}M)")
# Paper reports ~54M. We come in ~45.6M because our shift vocabulary is 1001 values
# (chained for longer, see §8) rather than the paper's 6001. That shrinks the token
# embedding and logits tables by ~5M params. Every other dimension matches the paper
# exactly (d_model=512, d_ff=1024, 6 heads, d_kv=64, 8 encoder + 8 decoder layers,
# GEGLU, sinusoidal positional embeddings).
print("(Paper: ~54M. Gap is from our max_shift_steps=1000 vs paper's 6000 — same info, fewer embeddings.)")


def shift_right(tokens: torch.Tensor, start_id: int = PAD_ID) -> torch.Tensor:
    """Standard seq2seq: prepend start-of-seq token, drop last — the T5 convention uses PAD as start."""
    shifted = torch.full_like(tokens, PAD_ID)
    shifted[:, 1:] = tokens[:, :-1]
    shifted[:, 0] = start_id
    return shifted


def compute_loss(logits: torch.Tensor, targets: torch.Tensor,
                 z_loss_coef: float = 0.0, label_smoothing: float = 0.0
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, V = logits.shape
    flat_logits = logits.reshape(B * L, V)
    flat_targets = targets.reshape(B * L)
    ce = F.cross_entropy(
        flat_logits, flat_targets,
        ignore_index=PAD_ID,
        label_smoothing=label_smoothing,
        reduction="sum",
    )
    n_real = (flat_targets != PAD_ID).sum().clamp_min(1)
    loss = ce / n_real

    if z_loss_coef > 0:
        # T5 z-loss (Z_LOSS = 1e-4 in mt3/gin/model.gin). Penalises large log-partition values
        # to keep the final softmax numerically stable during mixed-precision training.
        log_z = torch.logsumexp(flat_logits.float(), dim=-1)
        mask = (flat_targets != PAD_ID).float()
        z = (mask * log_z.pow(2)).sum() / n_real
        loss = loss + z_loss_coef * z

    with torch.no_grad():
        preds = flat_logits.argmax(-1)
        correct = ((preds == flat_targets) & (flat_targets != PAD_ID)).sum().float()
        acc = correct / n_real
    return loss, acc


_probe = DataLoader(train_ds, batch_size=2, num_workers=0)
_b = next(iter(_probe))
_inp = _b["inputs"].to(device)
_tgt = _b["targets"].to(device)
_di  = shift_right(_tgt)
with torch.no_grad():
    _logits = model(_inp, _di)
    _loss, _acc = compute_loss(_logits, _tgt, z_loss_coef=config.z_loss)
print(f"logits {tuple(_logits.shape)}   loss={_loss.item():.3f}   token-acc={_acc.item():.3f}")
# A freshly-initialised model on a 1280-class vocab should have loss ≈ ln(1280) ≈ 7.15.
print(f"Expected initial loss for uniform ~vocab size {VOCAB_SIZE_PADDED}: {math.log(VOCAB_SIZE_PADDED):.2f}")


optimizer = Adafactor(
    model.parameters(),
    lr=config.learning_rate,
    scale_parameter=False,   # don't scale learning rate by parameter norm
    relative_step=False,     # use the LR we pass in
    warmup_init=False,
    beta1=None,              # no first-moment estimate (Adafactor default)
    weight_decay=0.0,
    clip_threshold=1.0,      # internal Adafactor update-norm clip
)


def lr_at_step(step: int) -> float:
    """Linear warm-up then constant."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    return config.learning_rate


print(f"Optimizer: Adafactor, lr={config.learning_rate}, warmup={config.warmup_steps} steps")


train_loader = DataLoader(
    train_ds, batch_size=config.batch_size, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True,
)

# torch.amp is the non-deprecated API (torch >= 2.0). Wrap in a try for older installs.
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(config.use_amp and device.type == "cuda"))
    _amp_ctx = lambda: torch.amp.autocast("cuda", enabled=(config.use_amp and device.type == "cuda"), dtype=torch.float16)
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(config.use_amp and device.type == "cuda"))
    _amp_ctx = lambda: torch.cuda.amp.autocast(enabled=(config.use_amp and device.type == "cuda"), dtype=torch.float16)


def run_micro_batch(batch) -> Tuple[float, float]:
    inputs  = batch["inputs"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    dec_in  = shift_right(targets)
    with _amp_ctx():
        logits = model(inputs, dec_in)
        loss, acc = compute_loss(
            logits, targets,
            z_loss_coef=config.z_loss,
            label_smoothing=config.label_smoothing,
        )
        loss = loss / config.grad_accum
    scaler.scale(loss).backward()
    return loss.item() * config.grad_accum, acc.item()


history = {"step": [], "loss": [], "acc": [], "lr": []}
start_step = 0
latest_ckpt = CKPT_DIR / f"{config.input_type}_h{config.num_heads}_latest.pt"
if latest_ckpt.exists():
    print(f"Resuming from {latest_ckpt}")
    blob = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    start_step = blob["step"]
    history = blob.get("history", history)
    if "scaler" in blob:
        scaler.load_state_dict(blob["scaler"])

step = start_step
train_iter = iter(train_loader)
model.train()

pbar = tqdm(initial=start_step, total=config.total_steps, desc="train")
running_loss, running_acc, n_log = 0.0, 0.0, 0

while step < config.total_steps:
    optimizer.zero_grad(set_to_none=True)
    accum_loss, accum_acc = 0.0, 0.0
    for _ in range(config.grad_accum):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        l, a = run_micro_batch(batch)
        accum_loss += l; accum_acc += a
    accum_loss /= config.grad_accum
    accum_acc  /= config.grad_accum

    lr_now = lr_at_step(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_now

    # Unscale so we can clip; then take the step.
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    step += 1; pbar.update(1)
    running_loss += accum_loss; running_acc += accum_acc; n_log += 1

    if step % config.log_every == 0:
        avg_loss = running_loss / n_log
        avg_acc  = running_acc  / n_log
        history["step"].append(step); history["loss"].append(avg_loss)
        history["acc"].append(avg_acc); history["lr"].append(lr_now)
        pbar.set_postfix(loss=f"{avg_loss:.3f}", acc=f"{avg_acc:.3f}", lr=f"{lr_now:.1e}")
        running_loss, running_acc, n_log = 0.0, 0.0, 0

    if step % config.ckpt_every == 0 or step == config.total_steps:
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
        }, latest_ckpt)

pbar.close()
print("\nTraining complete.")


best_ckpt = CKPT_DIR / f"{config.input_type}_h{config.num_heads}_best.pt"
torch.save({"step": step, "model": model.state_dict(), "history": history}, best_ckpt)
print(f"Saved best checkpoint to {best_ckpt}")


@torch.no_grad()
def greedy_decode(model: PianoTranscriptionTransformer,
                  encoder_input: torch.Tensor,
                  max_length: Optional[int] = None) -> torch.Tensor:
    """encoder_input: [1, input_length, n_mels]. Returns predicted tokens [L_out] (no leading PAD)."""
    model.eval()
    if max_length is None:
        max_length = model.cfg.target_length
    encoded = model.encode(encoder_input)

    tokens = torch.full((1, 1), PAD_ID, device=encoder_input.device, dtype=torch.long)
    for _ in range(max_length - 1):
        logits = model.decode(tokens, encoded)           # [1, t+1, V]
        next_id = int(logits[0, -1].argmax().item())
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=tokens.device, dtype=torch.long)],
            dim=1,
        )
        if next_id == EOS_ID:
            break
    return tokens[0, 1:]     # drop the initial PAD (decoder-start)


def decode_events_to_notes(
        event_ids: List[int],
        codec: EventCodec,
        cfg: Config,
        time_offset: float = 0.0,
        initial_active: Optional[Dict[int, Tuple[float, int]]] = None,
) -> Tuple[List[Tuple[float, float, int, int]], Dict[int, Tuple[float, int]]]:
    current_time = time_offset
    current_vel_bin = 0
    active: Dict[int, Tuple[float, int]] = dict(initial_active or {})
    finished: List[Tuple[float, float, int, int]] = []
    chain_sum = 0

    for tok in event_ids:
        try:
            etype, val = codec.decode(tok)
        except ValueError:
            continue   # ignore malformed ids

        if etype == "shift":
            chain_sum += val
            continue

        if chain_sum > 0:
            # Replace current_time with segment-start + chain_sum/steps_per_second (NOT add).
            current_time = time_offset + chain_sum / cfg.steps_per_second
            chain_sum = 0

        if etype == "velocity":
            current_vel_bin = val
        elif etype == "pitch":
            pitch = val
            if current_vel_bin == 0:
                # Note-off.
                if pitch in active:
                    start_t, vel = active.pop(pitch)
                    if current_time > start_t:
                        finished.append((start_t, current_time, pitch, vel))
            else:
                # Note-on. If already active, close first, then restart.
                if pitch in active:
                    start_t, vel = active.pop(pitch)
                    if current_time > start_t:
                        finished.append((start_t, current_time, pitch, vel))
                active[pitch] = (current_time, bin_to_velocity(current_vel_bin, cfg.num_velocity_bins))
    return finished, active


@torch.no_grad()
def transcribe(model: PianoTranscriptionTransformer,
               audio: torch.Tensor,
               cfg: Config,
               codec: EventCodec,
               max_tokens_per_segment: int = 800,
               show_progress: bool = True,
              ) -> List[Tuple[float, float, int, int]]:
    model.eval()
    seg_samples = cfg.segment_samples
    n_segments = max(1, math.ceil(audio.numel() / seg_samples))

    all_notes: List[Tuple[float, float, int, int]] = []
    active: Dict[int, Tuple[float, int]] = {}

    rng = range(n_segments)
    if show_progress:
        rng = tqdm(rng, desc="transcribe", leave=False)

    for i in rng:
        s0 = i * seg_samples
        s1 = min(s0 + seg_samples, audio.numel())
        seg = audio[s0:s1]
        if seg.numel() < seg_samples:
            seg = F.pad(seg, (0, seg_samples - seg.numel()))

        mel = compute_spectrogram(seg)[:cfg.input_length]          # [512, n_mels]
        mel = mel.unsqueeze(0).to(device)                       # [1, 512, n_mels]

        pred_ids = greedy_decode(model, mel, max_length=max_tokens_per_segment).tolist()

        # Strip specials, convert model IDs back to event IDs.
        event_ids: List[int] = []
        for t in pred_ids:
            if t == EOS_ID:
                break
            if t < NUM_SPECIAL:
                continue
            eid = t - NUM_SPECIAL
            if 0 <= eid < codec.num_classes:
                event_ids.append(eid)

        seg_start_sec = s0 / cfg.sample_rate
        finished, active = decode_events_to_notes(
            event_ids, codec, cfg,
            time_offset=seg_start_sec,
            initial_active=active,
        )
        all_notes.extend(finished)

    # Close any still-active notes at the end of the audio.
    end_sec = audio.numel() / cfg.sample_rate
    for pitch, (start_t, vel) in active.items():
        if end_sec > start_t:
            all_notes.append((start_t, end_sec, pitch, vel))

    all_notes.sort()
    return all_notes


def _notes_to_arrays(notes):
    if not notes:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    intervals = np.array([[n[0], n[1]] for n in notes])
    pitch_hz  = 440.0 * (2.0 ** ((np.array([n[2] for n in notes]) - 69) / 12.0))
    velocity  = np.array([n[3] for n in notes])
    return intervals, pitch_hz, velocity


def evaluate_piece(model, piece_cache: Dict, cfg: Config, codec: EventCodec,
                   duration_sec: Optional[float] = 30.0) -> Dict[str, float]:
    audio = load_audio_16k(DATA_DIR / piece_cache["piece"]["audio_filename"])
    if duration_sec is not None:
        audio = audio[: int(duration_sec * cfg.sample_rate)]

    pred_notes = transcribe(model, audio, cfg, codec, show_progress=False)

    gt_notes = piece_cache["notes"]
    if duration_sec is not None:
        gt_notes = [
            (s, min(e, duration_sec), p, v) for s, e, p, v in gt_notes
            if s < duration_sec
        ]

    pred_iv, pred_hz, pred_v = _notes_to_arrays(pred_notes)
    gt_iv,   gt_hz,   gt_v   = _notes_to_arrays(gt_notes)

    if len(gt_iv) == 0 or len(pred_iv) == 0:
        zeros = {"P": 0.0, "R": 0.0, "F": 0.0}
        return {
            "onset":   zeros, "onoff": zeros, "onoffvel": zeros,
            "num_pred": len(pred_notes), "num_gt": len(gt_notes),
        }

    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_hz, pred_iv, pred_hz, offset_ratio=None)
    p_off, r_off, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_iv, gt_hz, pred_iv, pred_hz)
    p_vel, r_vel, f_vel, _ = mir_eval.transcription_velocity.precision_recall_f1_overlap(
        gt_iv, gt_hz, gt_v, pred_iv, pred_hz, pred_v)

    return {
        "onset":    {"P": p_on,  "R": r_on,  "F": f_on},
        "onoff":    {"P": p_off, "R": r_off, "F": f_off},
        "onoffvel": {"P": p_vel, "R": r_vel, "F": f_vel},
        "num_pred": len(pred_notes), "num_gt": len(gt_notes),
    }


print("Evaluating on validation set (first 30 s of each piece)...")
results = []
for pc in val_cache:
    r = evaluate_piece(model, pc, config, codec, duration_sec=30.0)
    title = pc["piece"]["canonical_title"][:45]
    print(f"  {title:45s}"
          f"  onset F1 {r['onset']['F']:.3f}"
          f"  on/off F1 {r['onoff']['F']:.3f}"
          f"  on/off/vel F1 {r['onoffvel']['F']:.3f}"
          f"  (pred {r['num_pred']}, gt {r['num_gt']})")
    results.append(r)

mean = lambda key, sub: float(np.mean([r[key][sub] for r in results]))
print("\nMean over validation set:")
print(f"  onset       P {mean('onset','P'):.3f}  R {mean('onset','R'):.3f}  F {mean('onset','F'):.3f}")
print(f"  onset+off   P {mean('onoff','P'):.3f}  R {mean('onoff','R'):.3f}  F {mean('onoff','F'):.3f}")
print(f"  on+off+vel  P {mean('onoffvel','P'):.3f}  R {mean('onoffvel','R'):.3f}  F {mean('onoffvel','F'):.3f}")


