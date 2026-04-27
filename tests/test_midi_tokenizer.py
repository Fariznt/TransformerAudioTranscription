"""
Unit tests for MT3Tokenizer.

All tests use synthetic PrettyMIDI objects so no MAPS dataset is required.
The expected token IDs are computed from the vocabulary layout in MT3_INTERNALS.md §1
and can be verified by hand against the constants in the tokenizer class.
"""
import pretty_midi
import pytest

from src.data.midi_tokenizer import MT3Tokenizer


@pytest.fixture
def tok():
    return MT3Tokenizer()


def make_midi(notes):
    """
    Build a PrettyMIDI object from a list of dicts:
      {program, pitch, velocity, start, end, is_drum=False}
    All non-drum notes share one instrument per program; drums share a single
    drum instrument (program=0, is_drum=True).
    """
    pm = pretty_midi.PrettyMIDI()
    instruments = {}  # (program, is_drum) -> Instrument
    for n in notes:
        is_drum = n.get("is_drum", False)
        prog    = n["program"]
        key     = (prog, is_drum)
        if key not in instruments:
            inst = pretty_midi.Instrument(program=prog, is_drum=is_drum)
            instruments[key] = inst
            pm.instruments.append(inst)
        instruments[key].notes.append(
            pretty_midi.Note(
                velocity=n["velocity"],
                pitch=n["pitch"],
                start=n["start"],
                end=n["end"],
            )
        )
    return pm


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class TestVocab:
    def test_vocab_size(self, tok):
        assert tok.vocab_size == 1517

    def test_special_tokens(self, tok):
        assert tok.PAD == 0
        assert tok.EOS == 1
        assert tok.UNK == 2

    def test_shift_token_bounds(self, tok):
        # shift(0) = 3, shift(1000) = 1003
        assert tok._SHIFT_BASE == 3
        assert tok._SHIFT_BASE + tok.MAX_SHIFT_STEPS == 1003

    def test_event_token_bases(self, tok):
        assert tok._PITCH_BASE == 1004
        assert tok._VEL_BASE   == 1132
        assert tok._TIE        == 1260
        assert tok._PROG_BASE  == 1261
        assert tok._DRUM_BASE  == 1389


# ---------------------------------------------------------------------------
# Encode — single piano note
# ---------------------------------------------------------------------------

class TestEncodeSingleNote:
    """
    One piano note: program=0, pitch=60 (middle C), velocity=64, t=0.5-1.0 s.
    Expected token stream (segment [0.0, 4.0]):

      Onset at 0.5 s = step 50:
        shift(50) = 50 + 3 = 53
        program(0) = 0 + 1261 = 1261
        velocity(64) = 64 + 1132 = 1196   [vel_bin(64) = ceil(127*64/127) = 64]
        pitch(60) = 60 + 1004 = 1064

      Offset at 1.0 s = step 100 (delta = 50):
        shift(50) = 53
        program(0) = 1261
        velocity(0) = 0 + 1132 = 1132    [note-off]
        pitch(60) = 1064

      EOS = 1
    """

    @pytest.fixture
    def tokens(self, tok):
        pm = make_midi([{"program": 0, "pitch": 60, "velocity": 64, "start": 0.5, "end": 1.0}])
        return tok.encode(pm, 0.0, 4.0)

    def test_no_eos_in_raw_encode(self, tokens):
        assert MT3Tokenizer.EOS not in tokens

    def test_onset_shift(self, tokens):
        # shift(50) should be first token
        assert tokens[0] == MT3Tokenizer._SHIFT_BASE + 50  # = 53

    def test_onset_program(self, tokens):
        assert tokens[1] == MT3Tokenizer._PROG_BASE + 0  # = 1261

    def test_onset_velocity(self, tokens):
        assert tokens[2] == MT3Tokenizer._VEL_BASE + 64  # = 1196

    def test_onset_pitch(self, tokens):
        assert tokens[3] == MT3Tokenizer._PITCH_BASE + 60  # = 1064

    def test_offset_shift(self, tokens):
        # Offset is at absolute step 100 (1.0 s); shift token encodes absolute position
        assert tokens[4] == MT3Tokenizer._SHIFT_BASE + 100

    def test_offset_velocity_is_zero(self, tokens):
        # velocity=0 encodes note-off
        assert tokens[6] == MT3Tokenizer._VEL_BASE + 0  # = 1132

    def test_total_length(self, tokens):
        # 1 shift + 3 onset + 1 shift + 3 offset = 8 tokens (EOS added by dataset layer)
        assert len(tokens) == 8


# ---------------------------------------------------------------------------
# Encode — drum hit
# ---------------------------------------------------------------------------

class TestEncodeDrum:
    def test_drum_tokens(self, tok):
        # Snare (pitch=38), velocity=100, t=0.1 s; segment [0, 4]
        pm = make_midi([{"program": 0, "pitch": 38, "velocity": 100,
                         "start": 0.1, "end": 0.15, "is_drum": True}])
        tokens = tok.encode(pm, 0.0, 4.0)

        # shift(10) = 13
        assert tokens[0] == tok._SHIFT_BASE + 10
        # velocity token (vel_bin(100) = ceil(127*100/127) = 100)
        assert tokens[1] == tok._VEL_BASE + 100
        # drum pitch token
        assert tokens[2] == tok._DRUM_BASE + 38

    def test_drum_has_no_offset_token(self, tok):
        # Drums must never produce a note-off event
        pm = make_midi([{"program": 0, "pitch": 38, "velocity": 100,
                         "start": 0.1, "end": 0.15, "is_drum": True}])
        tokens = tok.encode(pm, 0.0, 4.0)
        # Only 3 tokens: shift + vel + drum (EOS added by dataset layer)
        assert len(tokens) == 3


# ---------------------------------------------------------------------------
# Encode — boundary filtering
# ---------------------------------------------------------------------------

class TestEncodeBoundary:
    def test_empty_segment_returns_empty(self, tok):
        pm = make_midi([{"program": 0, "pitch": 60, "velocity": 64,
                         "start": 5.0, "end": 6.0}])
        tokens = tok.encode(pm, 0.0, 4.0)
        assert tokens == []

    def test_note_starting_before_segment_has_no_onset(self, tok):
        # Note starts at -0.5 s (before segment), ends at 1.0 s (inside segment).
        # Only a note-off should be emitted; no onset.
        pm = make_midi([{"program": 0, "pitch": 60, "velocity": 64,
                         "start": -0.5, "end": 1.0}])
        tokens = tok.encode(pm, 0.0, 4.0)
        # Should contain velocity(0) (note-off) but not a note-on velocity
        vel_zero = tok._VEL_BASE + 0
        vel_on   = tok._VEL_BASE + tok._vel_to_bin(64)
        assert vel_zero in tokens
        assert vel_on not in tokens

    def test_note_ending_after_segment_has_no_offset(self, tok):
        # Note starts at 1.0 s (inside), ends at 5.0 s (outside segment [0, 4]).
        # Only an onset should be emitted; no note-off.
        pm = make_midi([{"program": 0, "pitch": 60, "velocity": 64,
                         "start": 1.0, "end": 5.0}])
        tokens = tok.encode(pm, 0.0, 4.0)
        vel_zero = tok._VEL_BASE + 0
        assert vel_zero not in tokens  # no note-off emitted


# ---------------------------------------------------------------------------
# Encode — offset-before-onset ordering at the same timestamp
# ---------------------------------------------------------------------------

class TestEncodeOrdering:
    def test_offset_before_onset_at_same_time(self, tok):
        # Note A ends at t=1.0; note B starts at t=1.0.
        # The offset for A should appear before the onset for B.
        pm = make_midi([
            {"program": 0, "pitch": 60, "velocity": 64, "start": 0.5, "end": 1.0},
            {"program": 0, "pitch": 64, "velocity": 80, "start": 1.0, "end": 1.5},
        ])
        tokens = tok.encode(pm, 0.0, 4.0)

        # Find positions of velocity=0 (offset) and velocity=80 (onset of B)
        vel_off   = tok._VEL_BASE + 0
        vel_on_b  = tok._VEL_BASE + tok._vel_to_bin(80)
        idx_off   = tokens.index(vel_off)
        idx_on_b  = tokens.index(vel_on_b)
        assert idx_off < idx_on_b


# ---------------------------------------------------------------------------
# Round-trip: encode → decode
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_single_note_round_trip(self, tok):
        pm_in = make_midi([{"program": 0, "pitch": 60, "velocity": 64,
                            "start": 0.5, "end": 1.0}])
        tokens = tok.encode(pm_in, 0.0, 4.0) + [MT3Tokenizer.EOS]
        pm_out = tok.decode(tokens)

        assert len(pm_out.instruments) == 1
        assert len(pm_out.instruments[0].notes) == 1

        note = pm_out.instruments[0].notes[0]
        assert note.pitch    == 60
        assert note.velocity == 64
        assert abs(note.start - 0.5) < 0.015   # within one 10 ms step
        assert abs(note.end   - 1.0) < 0.015

    def test_drum_round_trip(self, tok):
        pm_in = make_midi([{"program": 0, "pitch": 38, "velocity": 100,
                            "start": 0.1, "end": 0.15, "is_drum": True}])
        tokens = tok.encode(pm_in, 0.0, 4.0) + [MT3Tokenizer.EOS]
        pm_out = tok.decode(tokens)

        assert len(pm_out.instruments) == 1
        assert pm_out.instruments[0].is_drum
        note = pm_out.instruments[0].notes[0]
        assert note.pitch == 38
        assert abs(note.start - 0.1) < 0.015
