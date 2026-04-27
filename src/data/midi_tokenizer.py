"""
Role: MIDI sequence tokenization according to the MT3 paper.
Converts PrettyMIDI objects or note lists into a sequence of discrete events:
- TimeShift (events denoting advancement of time)
- NoteOn (pitch, velocity)
- NoteOff (pitch)
- Program (instrument selection)
And also handles decoding tokens back into a sequence of MIDI events.
"""
from typing import List
import pretty_midi

class MT3Tokenizer:
    """
    Handles vocabulary creation and encoding/decoding of MIDI tokens.
    """
    def __init__(self):
        """
        Initializes vocabulary indices for pad, eos, sos, shift, pitch, velocity, and program.
        """
        self.vocab_size = 0 # To be calculated based on bins and ranges
        pass

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
        return []

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """
        Converts a list of token integers back into a PrettyMIDI object.
        We need
        """
        pass
