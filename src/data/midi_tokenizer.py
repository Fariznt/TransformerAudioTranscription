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

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> List[int]:
        """
        Converts a PrettyMIDI object into a list of vocabulary token integers.
        Must handle quantization of time and velocity.
        """
        pass

    def decode(self, tokens: List[int]) -> pretty_midi.PrettyMIDI:
        """
        Converts a list of token integers back into a PrettyMIDI object.
        We need
        """
        pass
