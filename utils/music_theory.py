"""
utils/music_theory.py
─────────────────────
Music-theory primitives used across the system:

  • Note names ↔ MIDI numbers
  • Scale / mode generation
  • Chord vocabulary & voicings
  • Chord progression analysis helpers
  • Harmony scoring utilities

All functions are pure Python / NumPy (no audio dependency).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ─── Note name tables ────────────────────────────────────────────────────────

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B",
}

# ─── MIDI helpers ────────────────────────────────────────────────────────────

def midi_to_note_name(midi: int) -> str:
    """60 → 'C4'"""
    octave = (midi // 12) - 1
    pc = PITCH_CLASSES[midi % 12]
    return f"{pc}{octave}"


def note_name_to_midi(name: str) -> int:
    """'C4' → 60,  'F#3' → 54"""
    # Parse octave
    if name[-1].isdigit() or (len(name) >= 2 and name[-2] == "-"):
        octave_str = name[-1] if name[-2] != "-" else name[-2:]
        pc_name = name[: len(name) - len(octave_str)]
    else:
        raise ValueError(f"Cannot parse note name: {name!r}")

    pc_name = ENHARMONIC.get(pc_name, pc_name)
    if pc_name not in PITCH_CLASSES:
        raise ValueError(f"Unknown pitch class: {pc_name!r}")

    pc = PITCH_CLASSES.index(pc_name)
    octave = int(octave_str)
    return (octave + 1) * 12 + pc


def pitch_class(midi: int) -> int:
    """Return pitch class 0-11 for a MIDI note."""
    return midi % 12


def pitch_class_name(midi: int) -> str:
    return PITCH_CLASSES[midi % 12]


# ─── Scale definitions ───────────────────────────────────────────────────────

SCALE_INTERVALS: Dict[str, List[int]] = {
    "major":              [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":      [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":     [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor_asc":  [0, 2, 3, 5, 7, 9, 11],
    "dorian":             [0, 2, 3, 5, 7, 9, 10],
    "phrygian":           [0, 1, 3, 5, 7, 8, 10],
    "lydian":             [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":         [0, 2, 4, 5, 7, 9, 10],
    "locrian":            [0, 1, 3, 5, 6, 8, 10],
    "pentatonic_major":   [0, 2, 4, 7, 9],
    "pentatonic_minor":   [0, 3, 5, 7, 10],
    "blues":              [0, 3, 5, 6, 7, 10],
    "chromatic":          list(range(12)),
}


def get_scale(root_midi: int, scale_name: str = "major") -> List[int]:
    """
    Return MIDI note numbers for one octave of the requested scale.

    Parameters
    ----------
    root_midi : root note (e.g. 60 for C4)
    scale_name: key into SCALE_INTERVALS

    Returns
    -------
    list of MIDI note numbers starting from root
    """
    intervals = SCALE_INTERVALS[scale_name]
    return [root_midi + i for i in intervals]


def scale_degree(midi: int, root_midi: int, scale_name: str = "major") -> Optional[int]:
    """
    Return 0-based scale degree of midi relative to root, or None if not in scale.
    """
    intervals = SCALE_INTERVALS[scale_name]
    pc_diff = (midi - root_midi) % 12
    if pc_diff in intervals:
        return intervals.index(pc_diff)
    return None


# ─── Chord definitions ───────────────────────────────────────────────────────

CHORD_INTERVALS: Dict[str, List[int]] = {
    "maj":   [0, 4, 7],
    "min":   [0, 3, 7],
    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],
    "sus2":  [0, 2, 7],
    "sus4":  [0, 5, 7],
    "maj7":  [0, 4, 7, 11],
    "min7":  [0, 3, 7, 10],
    "dom7":  [0, 4, 7, 10],
    "dim7":  [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "min_maj7": [0, 3, 7, 11],
    "maj6":  [0, 4, 7, 9],
    "min6":  [0, 3, 7, 9],
    "9":     [0, 4, 7, 10, 14],
    "maj9":  [0, 4, 7, 11, 14],
    "min9":  [0, 3, 7, 10, 14],
    "add9":  [0, 4, 7, 14],
}

# Chord vocabulary index (for tokenization)
CHORD_TYPES = list(CHORD_INTERVALS.keys())     # 18 quality names
# Full chord vocab = 12 roots × 18 types + 1 (no-chord) = 217 tokens
NO_CHORD_TOKEN = len(PITCH_CLASSES) * len(CHORD_TYPES)   # index 216

CHORD_TO_IDX: Dict[Tuple[int, str], int] = {
    (root, ctype): root * len(CHORD_TYPES) + i
    for root in range(12)
    for i, ctype in enumerate(CHORD_TYPES)
}
IDX_TO_CHORD: Dict[int, Tuple[int, str]] = {v: k for k, v in CHORD_TO_IDX.items()}


def chord_vocab_size() -> int:
    return NO_CHORD_TOKEN + 1


def get_chord_notes(root_midi: int, chord_type: str, inversion: int = 0) -> List[int]:
    """
    Return MIDI note numbers for a chord.

    Parameters
    ----------
    root_midi  : root note (e.g. 60 for C4)
    chord_type : key into CHORD_INTERVALS
    inversion  : 0 = root position, 1 = 1st inversion, …

    Returns
    -------
    list of MIDI note numbers
    """
    intervals = CHORD_INTERVALS[chord_type]
    notes = [root_midi + i for i in intervals]
    if inversion > 0:
        inv = inversion % len(notes)
        notes = notes[inv:] + [n + 12 for n in notes[:inv]]
    return notes


def chord_to_token(root_pc: int, chord_type: str) -> int:
    """Encode (root_pc, chord_type) as a single integer token."""
    return CHORD_TO_IDX.get((root_pc % 12, chord_type), NO_CHORD_TOKEN)


def token_to_chord(token: int) -> Tuple[Optional[int], Optional[str]]:
    """Decode a chord token back to (root_pc, chord_type). Returns (None,None) for no-chord."""
    if token == NO_CHORD_TOKEN:
        return None, None
    pair = IDX_TO_CHORD.get(token)
    if pair is None:
        return None, None
    return pair


def chord_name(root_pc: int, chord_type: str) -> str:
    """'C maj', 'F# min7', …"""
    return f"{PITCH_CLASSES[root_pc % 12]} {chord_type}"


# ─── Diatonic chord progressions ─────────────────────────────────────────────

# (scale_degree 0-6) → (chord_type, roman_numeral) for major key
MAJOR_DIATONIC = {
    0: ("maj",   "I"),
    1: ("min",   "ii"),
    2: ("min",   "iii"),
    3: ("maj",   "IV"),
    4: ("dom7",  "V7"),
    5: ("min",   "vi"),
    6: ("dim",   "vii°"),
}

MINOR_DIATONIC = {
    0: ("min",   "i"),
    1: ("dim",   "ii°"),
    2: ("maj",   "III"),
    3: ("min",   "iv"),
    4: ("min",   "v"),    # natural minor; use dom7 for harmonic minor
    5: ("maj",   "VI"),
    6: ("maj",   "VII"),
}

# Common pop / jazz progressions (as lists of degree indices, major key)
COMMON_PROGRESSIONS = {
    "I–V–vi–IV":  [0, 4, 5, 3],
    "I–IV–V–I":   [0, 3, 4, 0],
    "ii–V–I":     [1, 4, 0],
    "I–vi–IV–V":  [0, 5, 3, 4],
    "I–I–IV–V":   [0, 0, 3, 4],
    "vi–IV–I–V":  [5, 3, 0, 4],
    "I–V–IV–IV":  [0, 4, 3, 3],
    "blues_12bar": [0, 0, 0, 0, 3, 3, 0, 0, 4, 3, 0, 4],
}


def diatonic_chords(root_midi: int, scale_name: str = "major") -> List[Dict]:
    """
    Return all diatonic triads/seventh chords for the given root + scale.

    Returns
    -------
    list of dicts: {"degree": int, "root_midi": int, "root_pc": int,
                    "chord_type": str, "roman": str, "notes": List[int]}
    """
    scale_notes = get_scale(root_midi, scale_name)
    diatonic = MAJOR_DIATONIC if "major" in scale_name else MINOR_DIATONIC
    chords = []
    for degree, (ctype, roman) in diatonic.items():
        if degree < len(scale_notes):
            r = scale_notes[degree]
            chords.append(
                {
                    "degree":     degree,
                    "root_midi":  r,
                    "root_pc":    r % 12,
                    "chord_type": ctype,
                    "roman":      roman,
                    "notes":      get_chord_notes(r, ctype),
                }
            )
    return chords


# ─── Harmony scoring ─────────────────────────────────────────────────────────

def melody_chord_compatibility(
    melody_midis: List[int],
    chord_root_pc: int,
    chord_type: str,
    weight_chord_tones: float = 2.0,
    weight_scale_tones: float = 1.0,
) -> float:
    """
    Score how well a melody fragment fits a chord.

    Chord tones score higher than diatonic non-chord tones;
    non-diatonic tones score 0.
    """
    chord_pcs = set(
        (chord_root_pc + i) % 12
        for i in CHORD_INTERVALS.get(chord_type, [])
    )
    score = 0.0
    for m in melody_midis:
        pc = m % 12
        if pc in chord_pcs:
            score += weight_chord_tones
        else:
            score += 0.0
    return score / max(len(melody_midis), 1)


def key_from_notes(midi_notes: List[int]) -> Tuple[int, str]:
    """
    Estimate key (root_pc, 'major'/'minor') using Krumhansl-Schmuckler profiles.

    Returns the best-matching (root_pc, scale_name) pair.
    """
    # Krumhansl-Schmuckler key profiles
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    # Build chromagram from MIDI notes
    chroma = np.zeros(12)
    for m in midi_notes:
        chroma[m % 12] += 1
    if chroma.sum() == 0:
        return 0, "major"
    chroma = chroma / chroma.sum()

    best_score = -np.inf
    best_key = (0, "major")
    for root in range(12):
        for profile, sname in [(major_profile, "major"), (minor_profile, "natural_minor")]:
            rotated = np.roll(profile, root)
            score = float(np.corrcoef(chroma, rotated / rotated.sum())[0, 1])
            if score > best_score:
                best_score = score
                best_key = (root, sname)

    return best_key


# ─── MIDI note ↔ token ───────────────────────────────────────────────────────

MIDI_RANGE = (21, 108)          # 88-key piano
MELODY_VOCAB_SIZE = 130         # 128 MIDI + SOS(128) + EOS(129)
SOS_TOKEN = 128
EOS_TOKEN = 129
PAD_TOKEN = -1


def midi_to_token(midi: int) -> int:
    """Map MIDI note 0-127 directly to token id 0-127."""
    return max(0, min(127, int(midi)))


def token_to_midi(token: int) -> int:
    return max(0, min(127, token))


# ─── Example ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Scale (C major) ===")
    for m in get_scale(60, "major"):
        print(f"  {midi_to_note_name(m)} ({m})")

    print("\n=== Diatonic chords (G major) ===")
    for c in diatonic_chords(67, "major"):
        names = [midi_to_note_name(n) for n in c["notes"]]
        print(f"  {c['roman']:6s}  {chord_name(c['root_pc'], c['chord_type']):15s}  {names}")

    print("\n=== Key estimation ===")
    c_major_notes = get_scale(60, "major") * 2
    root, scale = key_from_notes(c_major_notes)
    print(f"  Estimated key: {PITCH_CLASSES[root]} {scale}")

    print("\n=== Chord token roundtrip ===")
    tok = chord_to_token(9, "min7")  # A minor 7
    r2, t2 = token_to_chord(tok)
    print(f"  A min7 → token {tok} → {PITCH_CLASSES[r2]} {t2}")
DRUM_MAP_ALIAS = {}
