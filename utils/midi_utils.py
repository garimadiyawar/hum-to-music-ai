"""
utils/midi_utils.py
───────────────────
Helpers for creating, reading, and manipulating MIDI data using
pretty_midi.  All public functions return / consume pretty_midi objects
or plain Python dicts so callers stay decoupled from the file format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pretty_midi

from utils.music_theory import (
    PITCH_CLASSES,
    get_chord_notes,
    token_to_chord,
    chord_name,
)

# ─── GM instrument presets ───────────────────────────────────────────────────

GM_PROGRAMS: Dict[str, int] = {
    # Pianos
    "acoustic_grand_piano": 0,
    "electric_piano":       4,
    # Chromatic percussion
    "vibraphone":          11,
    # Organ
    "hammond_organ":       16,
    # Strings
    "string_ensemble":     48,
    "violin":              40,
    "cello":               42,
    # Bass
    "acoustic_bass":       32,
    "electric_bass":       33,
    "synth_bass":          38,
    # Synth leads / pads
    "synth_lead":          80,
    "synth_pad":           88,
    "choir_aahs":          52,
    # Brass
    "trumpet":             56,
    "trombone":            57,
    # Guitar
    "nylon_guitar":        24,
    "electric_guitar":     26,
    # Woodwinds
    "flute":               73,
    "clarinet":            71,
}

# Named channel roles used by the arrangement model
ARRANGEMENT_TRACKS: Dict[str, Dict] = {
    "melody":  {"program": GM_PROGRAMS["acoustic_grand_piano"], "channel": 0, "is_drum": False},
    "bass":    {"program": GM_PROGRAMS["acoustic_bass"],         "channel": 1, "is_drum": False},
    "drums":   {"program": 0,                                    "channel": 9, "is_drum": True},
    "piano":   {"program": GM_PROGRAMS["electric_piano"],        "channel": 2, "is_drum": False},
    "strings": {"program": GM_PROGRAMS["string_ensemble"],       "channel": 3, "is_drum": False},
    "pad":     {"program": GM_PROGRAMS["synth_pad"],             "channel": 4, "is_drum": False},
}

# Drum map (GM standard channel 9)
DRUM_MAP: Dict[str, int] = {
    "kick":          36,
    "snare":         38,
    "hi_hat_closed": 42,
    "hi_hat_open":   46,
    "ride":          51,
    "crash":         49,
    "floor_tom":     43,
    "mid_tom":       47,
    "high_tom":      50,
}


# ─── Creation helpers ────────────────────────────────────────────────────────

def create_midi(tempo: float = 120.0, time_signature: Tuple[int, int] = (4, 4)) -> pretty_midi.PrettyMIDI:
    """
    Create an empty PrettyMIDI object with the specified tempo.
    """
    tempo = max(tempo, 60)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    return pm


def add_instrument(
    pm: pretty_midi.PrettyMIDI,
    program: int = 0,
    is_drum: bool = False,
    name: str = "Piano",
) -> pretty_midi.Instrument:
    """Add a new Instrument track and return it."""
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    pm.instruments.append(inst)
    return inst


def add_note(
    instrument: pretty_midi.Instrument,
    pitch: int,
    start: float,
    end: float,
    velocity: int = 80,
) -> None:
    """Append a single Note to an instrument track."""
    note = pretty_midi.Note(
        velocity=max(1, min(127, velocity)),
        pitch=max(0, min(127, pitch)),
        start=start,
        end=max(start + 0.01, end),
    )
    instrument.notes.append(note)


# ─── Note event list → MIDI ──────────────────────────────────────────────────

def notes_to_midi(
    note_events: List[Dict],
    tempo: float = 120.0,
    program: int = 0,
    instrument_name: str = "Melody",
    velocity: int = 80,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a list of note event dicts to a single-track PrettyMIDI object.

    note_events item format:
        {"pitch_midi": int, "start": float, "end": float, "velocity": int (optional)}
    """
    pm = create_midi(tempo=tempo)
    inst = add_instrument(pm, program=program, name=instrument_name)
    for ev in note_events:
        add_note(
            inst,
            pitch=int(ev["pitch_midi"]),
            start=float(ev["start"]),
            end=float(ev["end"]),
            velocity=int(ev.get("velocity", velocity)),
        )
    return pm


def chord_progression_to_midi(
    chords: List[Dict],
    tempo: float = 120.0,
    program: int = GM_PROGRAMS["electric_piano"],
    velocity: int = 65,
    voice_lead: bool = True,
) -> pretty_midi.PrettyMIDI:
    """
    Convert a chord progression to a piano MIDI track.

    chords item format:
        {"root_pc": int, "chord_type": str, "start": float, "end": float}
    """
    pm = create_midi(tempo=tempo)
    inst = add_instrument(pm, program=program, name="Harmony")
    prev_notes = None

    for chord in chords:
        root_midi = 48 + chord["root_pc"]   # octave 3 as base
        raw_notes = get_chord_notes(root_midi, chord["chord_type"])

        if voice_lead and prev_notes is not None:
            raw_notes = _voice_lead(prev_notes, raw_notes)

        for pitch in raw_notes:
            add_note(inst, pitch=pitch,
                     start=chord["start"], end=chord["end"],
                     velocity=velocity)
        prev_notes = raw_notes

    return pm


def _voice_lead(prev: List[int], curr: List[int]) -> List[int]:
    """Simple voice-leading: minimize total movement between chord voicings."""
    result = []
    for target in curr:
        # Find octave of target that is closest to the average of prev
        avg_prev = sum(prev) / len(prev)
        best = target
        best_dist = abs(target - avg_prev)
        for octave_shift in (-12, 0, 12, 24):
            candidate = target + octave_shift
            dist = abs(candidate - avg_prev)
            if dist < best_dist:
                best_dist = dist
                best = candidate
        result.append(best)
    return result


# ─── Multi-track arrangement → MIDI ─────────────────────────────────────────

def arrangement_to_midi(
    tracks: Dict[str, List[Dict]],
    tempo: float = 120.0,
) -> pretty_midi.PrettyMIDI:
    """
    Build a multi-track MIDI from an arrangement dict.

    Parameters
    ----------
    tracks : mapping of track_name → list[note_event dicts]
             Each note_event: {"pitch_midi": int, "start": float, "end": float,
                               "velocity": int (optional)}
    tempo  : BPM

    Returns
    -------
    pretty_midi.PrettyMIDI with one Instrument per track
    """
    pm = create_midi(tempo=tempo)
    for track_name, events in tracks.items():
        cfg = ARRANGEMENT_TRACKS.get(track_name, {"program": 0, "is_drum": False})
        inst = add_instrument(
            pm,
            program=cfg["program"],
            is_drum=cfg.get("is_drum", False),
            name=track_name.capitalize(),
        )
        for ev in events:
            add_note(
                inst,
                pitch=int(ev["pitch_midi"]),
                start=float(ev["start"]),
                end=float(ev["end"]),
                velocity=int(ev.get("velocity", 80)),
            )
    return pm


# ─── Reading MIDI ────────────────────────────────────────────────────────────

def load_midi(path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


def midi_to_note_events(pm: pretty_midi.PrettyMIDI) -> List[Dict]:
    """Flatten all instruments into a single list of note event dicts."""
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            events.append(
                {
                    "pitch_midi": note.pitch,
                    "start":      note.start,
                    "end":        note.end,
                    "duration":   note.end - note.start,
                    "velocity":   note.velocity,
                    "instrument": inst.name,
                    "program":    inst.program,
                    "is_drum":    inst.is_drum,
                }
            )
    events.sort(key=lambda e: e["start"])
    return events


def get_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    """Return first (or average) tempo in BPM."""
    tempos = pm.get_tempo_changes()
    if len(tempos[1]) == 0:
        return 120.0
    return float(np.mean(tempos[1]))


# ─── Piano roll helpers ──────────────────────────────────────────────────────

def midi_to_piano_roll(
    pm: pretty_midi.PrettyMIDI,
    fs: int = 100,
    pitch_range: Tuple[int, int] = (21, 108),
) -> np.ndarray:
    """
    Convert a PrettyMIDI object to a piano roll tensor.

    Returns
    -------
    ndarray shape (88, T)  – 88 piano keys × time steps
    """
    roll = pm.get_piano_roll(fs=fs)         # (128, T)
    roll = roll[pitch_range[0]: pitch_range[1] + 1, :]   # (88, T)
    return (roll > 0).astype(np.float32)


# ─── Token sequence helpers ──────────────────────────────────────────────────

def note_events_to_tokens(
    events: List[Dict],
    tempo: float = 120.0,
    ticks_per_beat: int = 480,
    velocity_bins: int = 32,
) -> List[int]:
    """
    Encode note events as a flat token sequence (MIDI-like event encoding).

    Token vocabulary:
        0-127   : NOTE_ON  (pitch)
        128-255 : NOTE_OFF (pitch)
        256-511 : TIME_SHIFT (256 × 10ms bins)
        512-543 : VELOCITY  (32 bins)

    This encoding is compatible with Music Transformer / PerformanceRNN.
    """
    NOTE_ON_OFFSET  = 0
    NOTE_OFF_OFFSET = 128
    TIME_OFFSET     = 256
    VEL_OFFSET      = 512
    TIME_BINS       = 256
    BIN_SIZE_SEC    = 0.01   # 10 ms per bin

    tokens = []
    current_time = 0.0

    # Expand to (onset, pitch, velocity) and (offset, pitch) events
    all_events = []
    for ev in events:
        all_events.append(("on",  ev["start"], ev["pitch_midi"], ev.get("velocity", 80)))
        all_events.append(("off", ev["end"],   ev["pitch_midi"], 0))

    all_events.sort(key=lambda x: (x[1], 0 if x[0] == "off" else 1))

    for ev in all_events:
        kind, t = ev[0], ev[1]
        dt = t - current_time
        # Emit time-shift tokens (up to 256 bins of 10 ms)
        while dt >= BIN_SIZE_SEC:
            bins = min(int(dt / BIN_SIZE_SEC), TIME_BINS - 1)
            tokens.append(TIME_OFFSET + bins)
            dt -= bins * BIN_SIZE_SEC
            current_time += bins * BIN_SIZE_SEC

        if kind == "on":
            pitch, vel = ev[2], ev[3]
            vel_bin = min(vel // (128 // velocity_bins), velocity_bins - 1)
            tokens.append(VEL_OFFSET + vel_bin)
            tokens.append(NOTE_ON_OFFSET + pitch)
        else:
            pitch = ev[2]
            tokens.append(NOTE_OFF_OFFSET + pitch)

    return tokens


# ─── Saving / Rendering ──────────────────────────────────────────────────────

def save_midi(pm: pretty_midi.PrettyMIDI, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
    print(f"[midi_utils] Saved MIDI → {path}")


def render_midi_to_audio(
    pm: pretty_midi.PrettyMIDI,
    soundfont_path: Optional[str] = None,
    sr: int = 44100,
) -> np.ndarray:
    """
    Render PrettyMIDI to a waveform using fluidsynth.

    Requires fluidsynth to be installed on the system
    and either a soundfont path or an environment variable SF2_PATH.

    Returns
    -------
    np.ndarray of float32 stereo audio, shape (samples, 2)
    """
    import os
    sf_path = soundfont_path or os.environ.get("SF2_PATH")
    if sf_path is None:
        raise RuntimeError(
            "No soundfont path provided.  "
            "Pass soundfont_path=... or set SF2_PATH environment variable."
        )
    audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
    return audio.astype(np.float32)


# ─── Example ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a simple C-major arpeggio
    notes = [
        {"pitch_midi": 60, "start": 0.0, "end": 0.5},
        {"pitch_midi": 64, "start": 0.5, "end": 1.0},
        {"pitch_midi": 67, "start": 1.0, "end": 1.5},
        {"pitch_midi": 72, "start": 1.5, "end": 2.0},
    ]
    pm = notes_to_midi(notes, tempo=120, instrument_name="Piano")

    chords = [
        {"root_pc": 0,  "chord_type": "maj",  "start": 0.0, "end": 2.0},
        {"root_pc": 5,  "chord_type": "maj",  "start": 2.0, "end": 4.0},
        {"root_pc": 7,  "chord_type": "dom7", "start": 4.0, "end": 6.0},
        {"root_pc": 0,  "chord_type": "maj",  "start": 6.0, "end": 8.0},
    ]
    pm_chords = chord_progression_to_midi(chords)

    save_midi(pm, "/tmp/arpeggio.mid")
    save_midi(pm_chords, "/tmp/chords.mid")

    # Token encoding demo
    tokens = note_events_to_tokens(notes)
    print(f"Token sequence length: {len(tokens)}, first 20: {tokens[:20]}")
