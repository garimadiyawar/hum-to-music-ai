"""
data/midi_processing.py
───────────────────────
MIDI file processing utilities used by dataset loaders and training pipelines.

Provides:
  • MidiParser    – parse MIDI → structured note events + chord annotations
  • MidiTokenizer – convert note events ↔ integer token sequences
  • Piano-roll extraction
  • Chord labelling heuristics from MIDI
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pretty_midi

from utils.music_theory import (
    PITCH_CLASSES,
    CHORD_TYPES,
    chord_to_token,
    token_to_chord,
    chord_vocab_size,
    NO_CHORD_TOKEN,
    key_from_notes,
    get_chord_notes,
    midi_to_note_name,
    SOS_TOKEN,
    EOS_TOKEN,
    MELODY_VOCAB_SIZE,
)
from utils.midi_utils import (
    load_midi,
    midi_to_note_events,
    get_tempo,
    note_events_to_tokens,
)


# ─── MIDI Parser ─────────────────────────────────────────────────────────────

class MidiParser:
    """
    Parse a MIDI file into structured representations.

    Attributes available after calling parse()
    ------------------------------------------
    note_events  : list[dict]  – all notes, sorted by onset
    melody_events: list[dict]  – highest-pitch (lead) track
    chord_events : list[dict]  – chord annotations extracted from polyphonic tracks
    tempo        : float
    key_root_pc  : int
    key_scale    : str
    duration     : float  (seconds)
    """

    def __init__(self, quantize_beats: bool = True, resolution: int = 480):
        self.quantize_beats = quantize_beats
        self.resolution = resolution

        self.note_events:   List[Dict] = []
        self.melody_events: List[Dict] = []
        self.chord_events:  List[Dict] = []
        self.tempo:         float = 120.0
        self.key_root_pc:   int   = 0
        self.key_scale:     str   = "major"
        self.duration:      float = 0.0

    def parse(self, path: Union[str, Path]) -> "MidiParser":
        """Parse a MIDI file. Returns self for chaining."""
        pm = load_midi(path)
        self.tempo = get_tempo(pm)
        self.duration = pm.get_end_time()
        self.note_events = midi_to_note_events(pm)

        midi_pitches = [e["pitch_midi"] for e in self.note_events if not e["is_drum"]]
        self.key_root_pc, self.key_scale = key_from_notes(midi_pitches) if midi_pitches else (0, "major")

        self._extract_melody()
        self._extract_chords(pm)
        return self

    # ── internal helpers ────────────────────────────────────────────────────

    def _extract_melody(self):
        """
        Heuristically extract the melody (highest-pitch, non-drum track).
        Uses a sliding-window max-pitch rule.
        """
        non_drum = [e for e in self.note_events if not e["is_drum"]]
        if not non_drum:
            self.melody_events = []
            return

        # Group by onset time (quantized to 50 ms)
        from collections import defaultdict
        bins: Dict[int, List[Dict]] = defaultdict(list)
        for e in non_drum:
            key = int(round(e["start"] / 0.05))
            bins[key].append(e)

        melody = []
        for k in sorted(bins):
            group = bins[k]
            # Highest pitch = melody note
            top = max(group, key=lambda x: x["pitch_midi"])
            melody.append(top)

        self.melody_events = melody

    def _extract_chords(self, pm: pretty_midi.PrettyMIDI, window: float = 0.5):
        """
        Extract chord annotations at regular intervals.
        Groups all simultaneously sounding notes within ``window`` seconds
        and finds the best-matching triad / seventh chord.
        """
        if self.duration <= 0:
            return

        times = np.arange(0, self.duration, window)
        chord_events = []

        for t in times:
            # Collect all pitches active in [t, t+window)
            active = [
                e["pitch_midi"]
                for e in self.note_events
                if not e["is_drum"] and e["start"] < t + window and e["end"] > t
            ]
            if not active:
                chord_events.append(
                    {"root_pc": None, "chord_type": None, "token": NO_CHORD_TOKEN,
                     "start": float(t), "end": float(t + window)}
                )
                continue

            root_pc, ctype, _ = _identify_chord(active)
            token = chord_to_token(root_pc, ctype) if root_pc is not None else NO_CHORD_TOKEN
            chord_events.append(
                {
                    "root_pc":    root_pc,
                    "chord_type": ctype,
                    "token":      token,
                    "start":      float(t),
                    "end":        float(t + window),
                }
            )

        self.chord_events = chord_events


def _identify_chord(
    pitches: List[int],
    min_score: float = 0.5,
) -> Tuple[Optional[int], Optional[str], float]:
    """
    Match a set of MIDI pitches to the best chord in the vocabulary.

    Returns (root_pc, chord_type, score).  score in [0, 1].
    """
    pcs = set(p % 12 for p in pitches)
    if not pcs:
        return None, None, 0.0

    from utils.music_theory import CHORD_INTERVALS
    best_score = -1.0
    best_root  = 0
    best_type  = "maj"

    for root in range(12):
        for ctype, intervals in CHORD_INTERVALS.items():
            chord_pcs = set((root + i) % 12 for i in intervals)
            if not chord_pcs:
                continue
            # Jaccard similarity
            intersection = len(pcs & chord_pcs)
            union = len(pcs | chord_pcs)
            score = intersection / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_root  = root
                best_type  = ctype

    if best_score < min_score:
        return None, None, best_score
    return best_root, best_type, best_score


# ─── MIDI Tokenizer ──────────────────────────────────────────────────────────

class MidiTokenizer:
    """
    Bidirectional tokenizer for MIDI note events.

    Vocabulary
    ----------
    0   … 127  : NOTE pitch (MIDI)
    128         : SOS
    129         : EOS
    130         : PAD
    """

    PAD_TOKEN = 130
    SOS_TOKEN = SOS_TOKEN     # 128
    EOS_TOKEN = EOS_TOKEN     # 129
    VOCAB_SIZE = 131

    def encode(
        self,
        note_events: List[Dict],
        add_sos: bool = True,
        add_eos: bool = True,
        max_len: Optional[int] = None,
    ) -> List[int]:
        """
        Encode a list of note event dicts to a token sequence.
        Order is by onset time; only pitch is encoded.
        """
        tokens = []
        if add_sos:
            tokens.append(self.SOS_TOKEN)

        pitches = [e["pitch_midi"] for e in sorted(note_events, key=lambda x: x["start"])]
        if max_len is not None:
            pitches = pitches[: max_len - (1 if add_sos else 0) - (1 if add_eos else 0)]

        tokens.extend(pitches)
        if add_eos:
            tokens.append(self.EOS_TOKEN)

        return tokens

    def decode(self, tokens: List[int]) -> List[int]:
        """
        Decode a token sequence back to a list of MIDI pitches.
        Strips SOS, EOS, PAD.
        """
        skip = {self.SOS_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN}
        return [t for t in tokens if t not in skip and 0 <= t <= 127]

    def pad(self, sequences: List[List[int]], max_len: Optional[int] = None) -> np.ndarray:
        """
        Pad a batch of token sequences to the same length.

        Returns
        -------
        np.ndarray of shape (B, max_len)
        """
        L = max_len or max(len(s) for s in sequences)
        out = np.full((len(sequences), L), fill_value=self.PAD_TOKEN, dtype=np.int64)
        for i, seq in enumerate(sequences):
            n = min(len(seq), L)
            out[i, :n] = seq[:n]
        return out


# ─── Piano roll helpers ──────────────────────────────────────────────────────

def extract_piano_roll(
    pm: pretty_midi.PrettyMIDI,
    fs: int = 100,
    pitch_low: int = 21,
    pitch_high: int = 108,
) -> np.ndarray:
    """
    Return binary piano roll of shape (pitch_high-pitch_low+1, T).
    """
    roll = pm.get_piano_roll(fs=fs)
    roll = roll[pitch_low : pitch_high + 1, :]
    return (roll > 0).astype(np.float32)


# ─── Melody token pair dataset helper ────────────────────────────────────────

def build_melody_chord_pairs(
    parser: MidiParser,
    bar_beats: int = 4,
    quantize: bool = True,
) -> List[Dict]:
    """
    Build (melody_tokens, chord_tokens) pairs from parsed MIDI.

    Each item: {"melody": List[int], "chords": List[int]}
    Bars are aligned to beat boundaries.
    """
    beats_per_sec = self_tempo_to_bps(parser.tempo)
    bar_dur = bar_beats / beats_per_sec

    if bar_dur <= 0 or parser.duration <= 0:
        return []

    pairs = []
    t = 0.0
    while t < parser.duration:
        t_end = t + bar_dur
        # Melody notes in this bar
        mel_notes = [
            e for e in parser.melody_events
            if e["start"] >= t and e["start"] < t_end
        ]
        mel_tokens = [e["pitch_midi"] for e in mel_notes]

        # Chord tokens in this bar (majority vote)
        chord_tokens_in_bar = [
            c["token"]
            for c in parser.chord_events
            if c["start"] >= t and c["start"] < t_end
        ]
        if chord_tokens_in_bar:
            # Use the most frequent chord token for the bar
            from collections import Counter
            chord_token = Counter(chord_tokens_in_bar).most_common(1)[0][0]
        else:
            chord_token = NO_CHORD_TOKEN

        if mel_tokens:
            pairs.append({"melody": mel_tokens, "chord": chord_token})
        t = t_end

    return pairs


def self_tempo_to_bps(tempo_bpm: float) -> float:
    return tempo_bpm / 60.0


# ─── MIDI file scanner ───────────────────────────────────────────────────────

def scan_midi_directory(
    root_dir: Union[str, Path],
    extensions: Tuple[str, ...] = (".mid", ".midi"),
    max_files: Optional[int] = None,
) -> List[Path]:
    """
    Recursively find all MIDI files under root_dir.

    Returns
    -------
    sorted list of Path objects
    """
    root = Path(root_dir)
    files = sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in extensions and p.is_file()
    )
    if max_files:
        files = files[:max_files]
    return files


# ─── Example ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Tokenizer demo ──
    tokenizer = MidiTokenizer()
    notes = [
        {"pitch_midi": 60, "start": 0.0, "end": 0.5},
        {"pitch_midi": 64, "start": 0.5, "end": 1.0},
        {"pitch_midi": 67, "start": 1.0, "end": 1.5},
    ]
    tokens = tokenizer.encode(notes)
    print("Encoded tokens:", tokens)
    pitches = tokenizer.decode(tokens)
    print("Decoded pitches:", pitches)

    # ── Chord identification demo ──
    c_major_pitches = [60, 64, 67]   # C E G
    root, ctype, score = _identify_chord(c_major_pitches)
    if root is not None:
        print(f"\nChord: {PITCH_CLASSES[root]} {ctype}  (score={score:.2f})")

    # ── Parse MIDI if provided ──
    if len(sys.argv) > 1:
        midi_path = sys.argv[1]
        parser = MidiParser()
        parser.parse(midi_path)
        print(f"\nParsed: {midi_path}")
        print(f"  Tempo: {parser.tempo:.1f} BPM")
        print(f"  Key: {PITCH_CLASSES[parser.key_root_pc]} {parser.key_scale}")
        print(f"  Duration: {parser.duration:.1f}s")
        print(f"  Notes: {len(parser.note_events)}")
        print(f"  Melody notes: {len(parser.melody_events)}")
        print(f"  Chord annotations: {len(parser.chord_events)}")
