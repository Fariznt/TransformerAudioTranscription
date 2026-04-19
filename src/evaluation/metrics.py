"""
Role: Wraps mir_eval functions to compute precision, recall, and F1 score for generated MIDI against ground truth.
Includes definitions for note-level and frame-level metrics.
"""
import mir_eval
import pretty_midi
from typing import Dict

def compute_note_metrics(pred_midi: pretty_midi.PrettyMIDI, true_midi: pretty_midi.PrettyMIDI, offset_ratio=0.2) -> Dict[str, float]:
    """
    Extracts intervals and pitches from both MIDI objects and uses mir_eval.transcription 
    to compute Precision, Recall, and F1.
    Includes Note On and Note On+Off metrics.
    """
    pass

def compute_multi_instrument_metrics(pred_midi, true_midi) -> Dict[str, float]:
    """
    If applicable, separates tracks by program/instrument and computes metrics per instrument,
    averaging them afterwards.
    """
    pass
