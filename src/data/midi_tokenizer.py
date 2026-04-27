"""
Role: MIDI sequence tokenization according to the MT3 paper.
Converts PrettyMIDI objects or note lists into a sequence of discrete events:
- TimeShift (events denoting advancement of time)
- NoteOn (pitch, velocity)
- NoteOff (pitch)
- Program (instrument selection)
And also handles decoding tokens back into a sequence of MIDI events.
"""
import math
from typing import List
import pretty_midi


class MT3Tokenizer:
    """
    Handles vocabulary creation and encoding/decoding of MIDI tokens.

    Vocabulary layout (matches MT3's GenericTokenVocabulary + build_codec exactly).
    The first 3 IDs are reserved special tokens; all event tokens are shifted by +3.
    See MT3_INTERNALS.md §1 for the full derivation.

      ID 0          PAD      – padding (ignored by loss)
      ID 1          EOS      – end of sequence
      ID 2          UNK      – unknown (not emitted; reserved for vocab compatibility)
      ID 3..1003    Shift    – advance time by (token - 3) steps; 100 steps = 1 second
      ID 1004..1131 Pitch    – MIDI note number 0-127  (token - 1004)
      ID 1132..1259 Velocity – 0 = note-off, 1-127 = velocity bin  (token - 1132)
      ID 1260       Tie      – segment-boundary sentinel (reserved; not emitted here)
      ID 1261..1388 Program  – MIDI program number 0-127  (token - 1261)
      ID 1389..1516 Drum     – MIDI drum pitch 0-127  (token - 1389)

    Total vocab size: 1517
    """

    # --- Special tokens ---
    PAD = 0
    EOS = 1
    UNK = 2

    # --- Temporal resolution ---
    # 100 steps/sec means each step = 10 ms, matching MT3's codec.
    # MAX_SHIFT_STEPS caps the largest single shift token (= 10 seconds).
    # Our segments are ~4 s so we'll never hit the cap, but it's part of the spec.
    STEPS_PER_SECOND = 100
    MAX_SHIFT_STEPS  = 1000

    # --- Velocity binning ---
    # MT3 reserves bin 0 for note-off; bins 1-127 map linearly onto MIDI velocity 1-127.
    # With 127 bins the mapping is effectively 1:1, but the indirection is kept so the
    # code matches the paper's formulas and can be changed if we experiment with coarser bins.
    NUM_VELOCITY_BINS = 127

    # --- Token ID base values (codec index + 3) ---
    _SHIFT_BASE = 3     # shift(n)    → token n + 3
    _PITCH_BASE = 1004  # pitch(p)    → token p + 1004
    _VEL_BASE   = 1132  # velocity(v) → token v + 1132
    _TIE        = 1260  # tie sentinel (not emitted by this encoder)
    _PROG_BASE  = 1261  # program(p)  → token p + 1261
    _DRUM_BASE  = 1389  # drum(d)     → token d + 1389

    VOCAB_SIZE = 1517

    def __init__(self):
        self.vocab_size = self.VOCAB_SIZE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _vel_to_bin(self, velocity: int) -> int:
        """Map a MIDI velocity (0-127) to a vocabulary bin index (0-127).
        Velocity 0 is reserved for note-off and maps to bin 0.
        Formula from mt3/vocabularies.py:63."""
        if velocity == 0:
            return 0
        return math.ceil(self.NUM_VELOCITY_BINS * velocity / 127)

    def _bin_to_vel(self, bin_val: int) -> int:
        """Inverse of _vel_to_bin: vocabulary bin index → MIDI velocity."""
        if bin_val == 0:
            return 0
        return round(bin_val * 127 / self.NUM_VELOCITY_BINS)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        midi_data: pretty_midi.PrettyMIDI,
        segment_start: float,
        segment_end: float,
    ) -> List[int]:
        """
        Convert the notes of ``midi_data`` that fall within
        ``[segment_start, segment_end]`` (seconds) into a list of vocabulary
        token integers, per Hawthorne et al. 2021 Sec. 3.2 / 3.3.

        Dev notes for future implementation:
        - Filter to notes whose [start, end] interval overlaps the segment.
        - Shift all event times by ``-segment_start`` so the segment begins at
          t = 0 (absolute time events are then quantized into 10 ms bins).
        - Emit a NoteOn only for notes that actually onset inside the segment
          (``note.start >= segment_start``). Notes inherited from a previous
          segment must NOT get a NoteOn -- the paper explicitly trains the
          model to predict their NoteOff without having seen the onset.
        - Emit a NoteOff for notes whose ``note.end`` falls inside the segment.
          Notes extending past ``segment_end`` get no NoteOff in this segment.
        - Quantize velocities into 128 bins; a zero-velocity NoteOn is
          interpreted as a NoteOff per the paper's vocabulary.
        """
        # ---- Step 1: collect events ----------------------------------------
        # We build two separate lists — offsets (note-off) first, onsets second —
        # then concatenate before sorting.  Python's sort is stable, so events at
        # the same timestamp will preserve offset-before-onset ordering.  This
        # matches MT3's note_sequence_to_onsets_and_offsets_and_programs.
        offset_events: list = []  # (time_s, [token, ...])
        onset_events:  list = []  # (time_s, [token, ...])

        # MT3 sorts notes by (is_drum, program, pitch) before iterating.
        instruments = sorted(midi_data.instruments, key=lambda i: (i.is_drum, i.program))
        for inst in instruments:
            prog    = inst.program
            is_drum = inst.is_drum

            for note in sorted(inst.notes, key=lambda n: n.pitch):
                # --- Offset (note-off) ---
                # Drums never get a note-off token; their duration is fixed at decode time.
                # A note-off is encoded as velocity=0 followed by the pitch token, preceded
                # by the program token (same layout as onset, just vel=0).
                if not is_drum and segment_start <= note.end <= segment_end:
                    offset_events.append((
                        note.end,
                        [self._PROG_BASE + prog, self._VEL_BASE + 0, self._PITCH_BASE + note.pitch],
                    ))

                # --- Onset (note-on) ---
                if segment_start <= note.start <= segment_end:
                    vel_bin = self._vel_to_bin(note.velocity)
                    if is_drum:
                        # Drums omit the program token; pitch goes to the drum range instead.
                        toks = [self._VEL_BASE + vel_bin, self._DRUM_BASE + note.pitch]
                    else:
                        toks = [self._PROG_BASE + prog, self._VEL_BASE + vel_bin, self._PITCH_BASE + note.pitch]
                    onset_events.append((note.start, toks))

        # ---- Step 2: sort by time ------------------------------------------
        all_events = offset_events + onset_events
        all_events.sort(key=lambda e: e[0])  # stable sort keeps offsets before onsets at equal times

        # ---- Step 3: emit tokens with RLE-compressed shifts ----------------
        # Time tokens encode ABSOLUTE position within the segment (paper Sec. 3.2),
        # not deltas from the previous event.  The MT3 decoder resets its step
        # accumulator to 0 after every non-shift token, so each event group must
        # be preceded by shift tokens that sum to its absolute step index.
        # Our segments are ≤ ~408 steps so a single shift token always suffices;
        # the while loop handles the general case for correctness.
        tokens: List[int] = []

        for (event_time, event_toks) in all_events:
            # Absolute step count from the beginning of the segment
            t = round((event_time - segment_start) * self.STEPS_PER_SECOND)
            t = max(0, min(t, self.MAX_SHIFT_STEPS))

            remaining = t
            while remaining > 0:
                s = min(remaining, self.MAX_SHIFT_STEPS)
                tokens.append(self._SHIFT_BASE + s)
                remaining -= s

            tokens.extend(event_toks)

        # EOS marks the end of the target sequence for the decoder.
        tokens.append(self.EOS)
        return tokens

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """
        Converts a list of token integers back into a PrettyMIDI object.
        We need this for evaluation: the model emits token IDs autoregressively,
        and we reconstruct MIDI to compare against the ground truth with mir_eval.

        Implements MT3's decode_events + decode_note_event state machine
        (mt3/run_length_encoding.py:371, mt3/note_sequences.py:313).
        """
        # Decoder state — updated token by token.
        # cur_steps mirrors MT3's decode_events accumulator: it resets to 0 after
        # every non-shift token so that the NEXT shift value re-encodes absolute
        # time from the segment start (matching the absolute-time encoding above).
        cur_steps    = 0
        cur_time     = 0.0   # seconds from segment start; updated by shift tokens
        cur_velocity = 0     # MIDI velocity for the next pitch/drum token (un-binned)
        cur_program  = 0     # MIDI program set by the most recent program token
        is_drum      = False # set to False by program tokens, implicitly True for drums

        # Open notes waiting for their note-off.
        # Key: (pitch, program, is_drum)  Value: (onset_time, onset_velocity)
        active: dict = {}

        # Completed notes ready to write into PrettyMIDI.
        # Each entry: (start_s, end_s, pitch, program, velocity, is_drum)
        completed: list = []

        for tok in tokens:
            # Stop at EOS or PAD; ignore UNK and Tie.
            if tok == self.EOS or tok == self.PAD:
                break

            if self._SHIFT_BASE <= tok <= self._SHIFT_BASE + self.MAX_SHIFT_STEPS:
                # Accumulate absolute steps; cur_time is measured from segment start.
                cur_steps += tok - self._SHIFT_BASE
                cur_time   = cur_steps / self.STEPS_PER_SECOND

            elif self._VEL_BASE <= tok <= self._VEL_BASE + 127:
                # Set velocity for the next pitch/drum token.
                # Velocity 0 means the following pitch closes the note (note-off).
                cur_velocity = self._bin_to_vel(tok - self._VEL_BASE)
                cur_steps = 0  # reset accumulator after non-shift token

            elif self._PITCH_BASE <= tok <= self._PITCH_BASE + 127:
                pitch = tok - self._PITCH_BASE
                key   = (pitch, cur_program, is_drum)
                if cur_velocity == 0:
                    # Note-off: close the matching open note.
                    if key in active:
                        onset_time, onset_vel = active.pop(key)
                        completed.append((onset_time, cur_time, pitch, cur_program, onset_vel, is_drum))
                else:
                    # Note-on: record the onset time and velocity.
                    active[key] = (cur_time, cur_velocity)
                cur_steps = 0  # reset accumulator after non-shift token

            elif self._PROG_BASE <= tok <= self._PROG_BASE + 127:
                # Select instrument for subsequent pitch tokens.
                cur_program = tok - self._PROG_BASE
                is_drum = False
                cur_steps = 0  # reset accumulator after non-shift token

            elif self._DRUM_BASE <= tok <= self._DRUM_BASE + 127:
                # Drum hits have no matching note-off; we assign a fixed 50 ms duration.
                pitch = tok - self._DRUM_BASE
                completed.append((cur_time, cur_time + 0.05, pitch, 0, cur_velocity, True))
                cur_steps = 0  # reset accumulator after non-shift token

        # Any notes still open at EOS get closed at the current time.
        for (pitch, prog, drum), (onset_time, onset_vel) in active.items():
            completed.append((onset_time, cur_time, pitch, prog, onset_vel, drum))

        # ---- Build PrettyMIDI object ----------------------------------------
        pm = pretty_midi.PrettyMIDI()
        instruments: dict = {}  # (program, is_drum) -> Instrument

        for (start, end, pitch, prog, vel, drum) in completed:
            key = (prog, drum)
            if key not in instruments:
                inst = pretty_midi.Instrument(program=prog, is_drum=drum)
                instruments[key] = inst
                pm.instruments.append(inst)
            note = pretty_midi.Note(
                velocity=max(1, vel),   # PrettyMIDI requires velocity >= 1
                pitch=pitch,
                start=start,
                end=max(end, start + 0.01),  # enforce a minimum 10 ms duration
            )
            instruments[key].notes.append(note)

        return pm
