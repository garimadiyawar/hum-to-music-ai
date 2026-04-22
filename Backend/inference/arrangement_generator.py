"""
inference/arrangement_generator.py
────────────────────────────────────
Stage 3 of the inference pipeline:

    melody + chords  →  multi-track MIDI arrangement

Two strategies:
  A) Neural:     ArrangementModel (requires trained checkpoint)
  B) Rule-based: pattern-based track generator (always available)

Public API
----------
    from inference.arrangement_generator import ArrangementGenerator

    gen = ArrangementGenerator(checkpoint="checkpoints/arrangement/best.pt")
    tracks = gen.arrange(notes, chords, tempo=120.0)   # → Dict[str, List[dict]]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.arrangement_model import (
    ArrangementModel,
    ARR_VOCAB_SIZE,
    ARR_SOS,
    TRACK_NAMES,
    decode_arrangement_tokens,
)
from models.composition_transformer import RuleBasedComposer, CompositionConfig
from utils.music_theory import (
    PITCH_CLASSES,
    get_chord_notes,
    chord_name,
    key_from_notes,
    DRUM_MAP_ALIAS,
)
from utils.midi_utils import ARRANGEMENT_TRACKS, DRUM_MAP


# Re-export DRUM_MAP alias for convenience
try:
    from utils.music_theory import DRUM_MAP_ALIAS
except ImportError:
    DRUM_MAP_ALIAS = {}


# ─── ArrangementGenerator ────────────────────────────────────────────────────

class ArrangementGenerator:
    """
    Generates a multi-track musical arrangement from melody + chords.

    Parameters
    ----------
    checkpoint   : path to ArrangementModel checkpoint (optional)
    device       : torch device
    temperature  : sampling temperature for neural model
    top_k        : nucleus filter top-k for neural model
    """

    def __init__(
        self,
        checkpoint:  Optional[str] = None,
        device:      str = "auto",
        temperature: float = 0.9,
        top_k:       int  = 50,
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda"  if torch.cuda.is_available()         else
                "mps"   if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        self.temperature = temperature
        self.top_k       = top_k
        self.model: Optional[ArrangementModel] = None

        if checkpoint and Path(checkpoint).exists():
            self._load_model(checkpoint)
        else:
            if checkpoint:
                print(f"[ArrangementGenerator] Checkpoint not found: {checkpoint}")
            print("[ArrangementGenerator] Using rule-based arrangement.")

    def _load_model(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg  = ckpt.get("model_config", {})
        self.model = ArrangementModel(
            d_model=cfg.get("d_model", 512),
            nhead=cfg.get("nhead", 8),
            num_enc_layers=cfg.get("num_enc_layers", 6),
            num_dec_layers=cfg.get("num_dec_layers", 6),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[ArrangementGenerator] Loaded ArrangementModel from {checkpoint}")

    # ── Public API ───────────────────────────────────────────────────────────

    def arrange(
        self,
        melody_notes: List[Dict],
        chords:       List[Dict],
        tempo:        float = 120.0,
        duration:     Optional[float] = None,
        key_root_pc:  int  = 0,
        key_scale:    str  = "major",
    ) -> Dict[str, List[Dict]]:
        """
        Generate a multi-track arrangement.

        Parameters
        ----------
        melody_notes : list of melody note event dicts
        chords       : list of chord dicts (from MelodyToChords)
        tempo        : BPM
        duration     : total duration in seconds
        key_root_pc  : key root pitch class (0-11)
        key_scale    : "major" | "natural_minor"

        Returns
        -------
        dict mapping track_name → list[note_event dict]
        Tracks: "melody", "bass", "drums", "piano", "strings", "pad"
        """
        if not melody_notes:
            return {name: [] for name in TRACK_NAMES}

        if duration is None:
            dur_list = [n["end"] for n in melody_notes] + [c["end"] for c in chords]
            duration = max(dur_list) if dur_list else 8.0

        if self.model is not None:
            tracks = self._arrange_neural(melody_notes, chords, tempo, duration)
        else:
            tracks = self._arrange_rules(melody_notes, chords, tempo, duration,
                                          key_root_pc, key_scale)

        # Always include the original melody track
        tracks["melody"] = melody_notes
        return tracks

    # ── Neural arrangement ────────────────────────────────────────────────────

    def _arrange_neural(
        self,
        melody_notes: List[Dict],
        chords:       List[Dict],
        tempo:        float,
        duration:     float,
    ) -> Dict[str, List[Dict]]:
        from data.midi_processing import MidiTokenizer
        tokenizer = MidiTokenizer()

        mel_tokens  = tokenizer.encode(melody_notes, add_sos=True, add_eos=True)
        chord_tokens = [c["token"] for c in chords]

        # Pad / truncate
        MAX_SRC = 128
        mel_tokens   = (mel_tokens  + [130] * MAX_SRC)[:MAX_SRC]
        chord_tokens = (chord_tokens + [0]   * MAX_SRC)[:MAX_SRC]

        mel_t   = torch.tensor([mel_tokens],   dtype=torch.long, device=self.device)
        chord_t = torch.tensor([chord_tokens], dtype=torch.long, device=self.device)

        arr_tokens = self.model.generate(
            mel_t, chord_t,
            max_len=512,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        tracks = decode_arrangement_tokens(arr_tokens)
        return tracks

    # ── Rule-based arrangement ────────────────────────────────────────────────

    def _arrange_rules(
        self,
        melody_notes: List[Dict],
        chords:       List[Dict],
        tempo:        float,
        duration:     float,
        key_root_pc:  int,
        key_scale:    str,
    ) -> Dict[str, List[Dict]]:
        """
        Pattern-based arrangement covering:
          bass, drums, piano (comp), strings (pad), synth pad
        """
        sec_per_beat = 60.0 / max(tempo, 1.0)
        tracks: Dict[str, List[Dict]] = {name: [] for name in TRACK_NAMES}

        for chord in chords:
            if chord["root_pc"] is None:
                continue
            start = chord["start"]
            end   = chord["end"]
            bar_dur = end - start

            # ── Bass (root + fifth, walking quarter notes) ─────────────────
            bass_root   = 36 + chord["root_pc"]   # bass octave (MIDI 36–48)
            bass_fifth  = bass_root + 7
            beat_dur    = sec_per_beat
            t = start
            beat_idx = 0
            while t < end:
                pitch = bass_root if beat_idx % 2 == 0 else bass_fifth
                note_dur = min(beat_dur * 0.9, end - t)
                if note_dur > 0.05:
                    tracks["bass"].append(
                        {"pitch_midi": pitch, "start": t, "end": t + note_dur, "velocity": 85}
                    )
                t += beat_dur
                beat_idx += 1

            # ── Piano (chord voicing on beats 1 & 3) ──────────────────────
            chord_pitches = get_chord_notes(48 + chord["root_pc"], chord["chord_type"])
            for beat_offset in [0.0, bar_dur / 2]:
                t_beat = start + beat_offset
                if t_beat >= end:
                    break
                for pitch in chord_pitches:
                    tracks["piano"].append(
                        {
                            "pitch_midi": min(84, pitch),
                            "start":      t_beat,
                            "end":        t_beat + min(sec_per_beat * 0.8, end - t_beat),
                            "velocity":   68,
                        }
                    )

            # ── Strings (sustained whole-bar chord, high voicing) ──────────
            for pitch in chord_pitches:
                tracks["strings"].append(
                    {
                        "pitch_midi": min(96, pitch + 12),
                        "start":      start,
                        "end":        end - 0.01,
                        "velocity":   55,
                    }
                )

            # ── Synth pad (root + octave, very soft) ──────────────────────
            tracks["pad"].append(
                {"pitch_midi": bass_root + 12, "start": start, "end": end - 0.01, "velocity": 40}
            )
            tracks["pad"].append(
                {"pitch_midi": bass_root + 24, "start": start, "end": end - 0.01, "velocity": 35}
            )

        # ── Drums (standard 4/4 rock/pop pattern) ─────────────────────────
        tracks["drums"] = _generate_drum_pattern(
            duration=duration,
            tempo=tempo,
            style="pop",
        )

        return tracks


# ─── Drum pattern generator ───────────────────────────────────────────────────

def _generate_drum_pattern(
    duration: float,
    tempo:    float,
    style:    str = "pop",
) -> List[Dict]:
    """
    Generate a standard drum pattern.

    Patterns (per bar):
        pop:   kick on 1&3, snare on 2&4, hi-hat every eighth
        jazz:  ride on every quarter, snare on 2&4, bass on 1
        hiphop: kick on 1&2.5&3, snare on 2&4
    """
    sec_per_beat  = 60.0 / max(tempo, 1.0)
    sec_per_eighth = sec_per_beat / 2

    kick   = DRUM_MAP["kick"]
    snare  = DRUM_MAP["snare"]
    hihat  = DRUM_MAP["hi_hat_closed"]
    ride   = DRUM_MAP["ride"]
    crash  = DRUM_MAP["crash"]

    def note(pitch, t, vel=80):
        return {"pitch_midi": pitch, "start": t, "end": t + 0.05, "velocity": vel}

    events = []

    # Add crash on bar 1
    events.append(note(crash, 0.0, vel=90))

    t = 0.0
    beat = 0
    while t < duration:
        bar_beat = beat % 4

        if style == "pop":
            # Kick: beats 0 and 2
            if bar_beat in (0, 2):
                events.append(note(kick,  t, vel=95))
            # Snare: beats 1 and 3
            if bar_beat in (1, 3):
                events.append(note(snare, t, vel=85))
            # Hi-hat: every eighth note
            events.append(note(hihat, t,                  vel=65))
            events.append(note(hihat, t + sec_per_eighth, vel=55))

        elif style == "jazz":
            events.append(note(ride, t, vel=75))
            if bar_beat in (1, 3):
                events.append(note(snare, t, vel=70))
            if bar_beat == 0:
                events.append(note(kick, t, vel=85))

        elif style == "hiphop":
            if bar_beat == 0:
                events.append(note(kick, t, vel=100))
            if bar_beat == 1:
                events.append(note(kick, t + sec_per_eighth, vel=90))
                events.append(note(snare, t, vel=90))
            if bar_beat == 2:
                events.append(note(kick, t, vel=95))
            if bar_beat == 3:
                events.append(note(snare, t, vel=88))
            # Hi-hat every eighth
            events.append(note(hihat, t,                  vel=60))
            events.append(note(hihat, t + sec_per_eighth, vel=50))

        t    += sec_per_beat
        beat += 1

    return events


# ─── Track statistics printer ─────────────────────────────────────────────────

def print_arrangement(tracks: Dict[str, List[Dict]]):
    """Pretty-print arrangement statistics."""
    print(f"\n{'Track':12}  {'Notes':>6}  {'Start':>6}  {'End':>6}")
    print("─" * 40)
    for name, events in tracks.items():
        if not events:
            print(f"{name:12}  {'(empty)':>6}")
            continue
        n = len(events)
        s = min(e["start"] for e in events)
        e_max = max(e["end"] for e in events)
        print(f"{name:12}  {n:6d}  {s:6.2f}  {e_max:6.2f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from utils.midi_utils import arrangement_to_midi, save_midi

    parser = argparse.ArgumentParser(description="Generate multi-track arrangement")
    parser.add_argument("melody_midi",  help="Input melody MIDI")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output",     default="outputs/arrangement.mid")
    parser.add_argument("--tempo",      type=float, default=120.0)
    args = parser.parse_args()

    # Load melody
    from utils.midi_utils import load_midi, midi_to_note_events
    pm    = load_midi(args.melody_midi)
    notes = [n for n in midi_to_note_events(pm) if not n["is_drum"]]
    print(f"Melody: {len(notes)} notes from {args.melody_midi}")

    # Harmonize
    from inference.melody_to_chords import MelodyToChords
    m2c    = MelodyToChords()
    chords = m2c.harmonize(notes, tempo=args.tempo)

    # Arrange
    gen    = ArrangementGenerator(checkpoint=args.checkpoint)
    tracks = gen.arrange(notes, chords, tempo=args.tempo)

    print_arrangement(tracks)

    # Save
    pm_out = arrangement_to_midi(tracks, tempo=args.tempo)
    save_midi(pm_out, args.output)
    print(f"\nSaved arrangement → {args.output}")
