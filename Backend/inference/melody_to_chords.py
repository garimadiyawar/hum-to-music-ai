"""
inference/melody_to_chords.py
──────────────────────────────
Stage 2 of the inference pipeline:

    MIDI note events  →  chord progression

Two strategies:
  A) Neural:     HarmonyGenerator (requires trained checkpoint)
  B) Rule-based: key-detection + diatonic chord heuristics (always available)

Public API
----------
    from inference.melody_to_chords import MelodyToChords

    m2c = MelodyToChords(checkpoint="checkpoints/harmony/best.pt")
    chords = m2c.harmonize(notes, duration=8.0, tempo=120.0)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.harmony_generator import HarmonyGenerator, MEL_PAD, MEL_VOCAB
from utils.music_theory import (
    PITCH_CLASSES,
    CHORD_TYPES,
    chord_to_token,
    token_to_chord,
    chord_name,
    chord_vocab_size,
    NO_CHORD_TOKEN,
    key_from_notes,
    diatonic_chords,
    COMMON_PROGRESSIONS,
    SOS_TOKEN,
    EOS_TOKEN,
    get_chord_notes,
)

CHORD_VOCAB = chord_vocab_size()


# ─── MelodyToChords ──────────────────────────────────────────────────────────

class MelodyToChords:
    """
    Harmonizes a melody (list of note events) into a chord progression.

    Parameters
    ----------
    checkpoint  : path to HarmonyGenerator checkpoint (optional)
    device      : torch device
    temperature : sampling temperature for neural model
    bar_beats   : beats per bar (default 4)
    """

    def __init__(
        self,
        checkpoint:  Optional[str] = None,
        device:      str = "auto",
        temperature: float = 0.9,
        bar_beats:   int = 4,
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
        self.bar_beats   = bar_beats
        self.model: Optional[HarmonyGenerator] = None

        if checkpoint and Path(checkpoint).exists():
            self._load_model(checkpoint)
        else:
            if checkpoint:
                print(f"[MelodyToChords] Checkpoint not found: {checkpoint}")
            print("[MelodyToChords] Using rule-based harmonization.")

    def _load_model(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg  = ckpt.get("model_config", {})
        self.model = HarmonyGenerator(
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 4),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[MelodyToChords] Loaded HarmonyGenerator from {checkpoint}")

    # ── Public API ───────────────────────────────────────────────────────────

    def harmonize(
        self,
        notes:    List[Dict],
        duration: Optional[float] = None,
        tempo:    float = 120.0,
        merge:    bool = True,
    ) -> List[Dict]:
        """
        Generate a chord progression for the given note events.

        Parameters
        ----------
        notes    : list of note event dicts
        duration : total duration in seconds (inferred from notes if None)
        tempo    : BPM (used for bar-length calculation)
        merge    : collapse adjacent identical chords

        Returns
        -------
        List of chord dicts:
            {"root_pc": int|None, "chord_type": str|None, "name": str,
             "token": int, "start": float, "end": float}
        """
        if not notes:
            return []

        if duration is None:
            duration = max(n["end"] for n in notes)

        if self.model is not None:
            return self._harmonize_neural(notes, duration, tempo, merge)
        else:
            return self._harmonize_rules(notes, duration, tempo)

    # ── Neural harmonization ─────────────────────────────────────────────────

    def _harmonize_neural(
        self,
        notes:    List[Dict],
        duration: float,
        tempo:    float,
        merge:    bool,
    ) -> List[Dict]:
        """
        Encode melody as token sequence and run HarmonyGenerator.
        """
        # Segment melody into bars and encode per bar
        sec_per_beat = 60.0 / max(tempo, 1.0)
        bar_dur      = self.bar_beats * sec_per_beat

        chords_out: List[Dict] = []
        t = 0.0

        while t < duration:
            t_end = t + bar_dur
            bar_notes = [n for n in notes if n["start"] >= t and n["start"] < t_end]

            if bar_notes:
                mel_tokens = [SOS_TOKEN] + [n["pitch_midi"] for n in bar_notes] + [EOS_TOKEN]
                # Truncate to model max_seq_len
                mel_tokens = mel_tokens[:64]
                # Pad
                pad_len = 64 - len(mel_tokens)
                mel_padded = mel_tokens + [MEL_PAD] * pad_len

                mel_t = torch.tensor([mel_padded], dtype=torch.long, device=self.device)
                chord_dicts = self.model.melody_to_chord_sequence(
                    mel_padded, merge_consecutive=False, temperature=self.temperature
                )

                # Use first predicted chord for this bar
                if chord_dicts:
                    c = chord_dicts[0]
                    chords_out.append({
                        "root_pc":    c["root_pc"],
                        "chord_type": c["chord_type"],
                        "name":       c["name"],
                        "token":      c["token"],
                        "start":      t,
                        "end":        t_end,
                    })
                else:
                    chords_out.append(_no_chord(t, t_end))
            else:
                # Silent bar: repeat previous chord or no-chord
                if chords_out:
                    prev = chords_out[-1]
                    chords_out.append({**prev, "start": t, "end": t_end})
                else:
                    chords_out.append(_no_chord(t, t_end))

            t = t_end

        if merge:
            chords_out = _merge_consecutive(chords_out)

        return chords_out

    # ── Rule-based harmonization ─────────────────────────────────────────────

    def _harmonize_rules(
        self,
        notes:    List[Dict],
        duration: float,
        tempo:    float,
    ) -> List[Dict]:
        """
        Key-aware diatonic harmonization using common chord progressions.
        """
        midi_pitches = [n["pitch_midi"] for n in notes]
        key_root, key_scale = key_from_notes(midi_pitches)

        # Get diatonic chord options for this key
        root_midi  = 60 + key_root   # octave 4
        diat_chords = diatonic_chords(root_midi, key_scale)

        # Choose a common progression
        prog_name   = _pick_progression(len(notes))
        prog_degrees = COMMON_PROGRESSIONS[prog_name]

        sec_per_beat = 60.0 / max(tempo, 1.0)
        bar_dur      = self.bar_beats * sec_per_beat

        chords_out: List[Dict] = []
        t          = 0.0
        prog_idx   = 0

        while t < duration:
            degree = prog_degrees[prog_idx % len(prog_degrees)]
            # Find the diatonic chord at this scale degree
            match = next(
                (c for c in diat_chords if c["degree"] == degree),
                diat_chords[0] if diat_chords else None,
            )
            if match:
                t_end = t + bar_dur
                token = chord_to_token(match["root_pc"], match["chord_type"])
                chords_out.append(
                    {
                        "root_pc":    match["root_pc"],
                        "chord_type": match["chord_type"],
                        "name":       chord_name(match["root_pc"], match["chord_type"]),
                        "token":      token,
                        "start":      t,
                        "end":        t_end,
                    }
                )
                t = t_end
            else:
                t += bar_dur
            prog_idx += 1

        return chords_out

    # ── Key & tempo analysis ──────────────────────────────────────────────────

    def analyze_melody(self, notes: List[Dict]) -> Dict:
        """
        Return key, scale, and recommended tempo from note events.

        Returns
        -------
        {"key_root_pc": int, "key_scale": str, "key_name": str,
         "note_count": int}
        """
        pitches = [n["pitch_midi"] for n in notes]
        root, scale = key_from_notes(pitches)
        return {
            "key_root_pc": root,
            "key_scale":   scale,
            "key_name":    f"{PITCH_CLASSES[root]} {scale}",
            "note_count":  len(notes),
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _no_chord(start: float, end: float) -> Dict:
    return {
        "root_pc":    None,
        "chord_type": None,
        "name":       "N.C.",
        "token":      NO_CHORD_TOKEN,
        "start":      start,
        "end":        end,
    }


def _merge_consecutive(chords: List[Dict]) -> List[Dict]:
    """Merge adjacent chords with the same token into one longer chord."""
    if not chords:
        return chords
    merged = [chords[0].copy()]
    for c in chords[1:]:
        if c["token"] == merged[-1]["token"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(c.copy())
    return merged


def _pick_progression(n_notes: int) -> str:
    """Pick a chord progression template based on melody length."""
    if n_notes <= 4:
        return "I–V–vi–IV"
    elif n_notes <= 8:
        return "I–IV–V–I"
    elif n_notes <= 12:
        return "I–vi–IV–V"
    else:
        return "I–V–vi–IV"


def print_chords(chords: List[Dict]):
    """Pretty-print a chord progression."""
    print(f"\n{'Bar':>3}  {'Name':15}  {'Start':>6}  {'End':>6}  Token")
    print("─" * 50)
    for i, c in enumerate(chords):
        print(
            f"{i+1:3d}  {c['name']:15}  "
            f"{c['start']:6.2f}  {c['end']:6.2f}  {c['token']}"
        )


def chords_to_note_events(
    chords:   List[Dict],
    velocity: int = 65,
    voicing:  str = "closed",   # "closed" | "open"
) -> List[Dict]:
    """
    Expand a chord list into individual note events (for MIDI playback).

    Returns
    -------
    list of note event dicts with pitch_midi, start, end, velocity
    """
    note_events = []
    for chord in chords:
        if chord["root_pc"] is None:
            continue
        root_midi = 48 + chord["root_pc"]      # octave 3
        pitches   = get_chord_notes(root_midi, chord["chord_type"])

        if voicing == "open" and len(pitches) >= 2:
            # Spread voicing: bass down an octave, inner voices spread
            pitches[0] -= 12

        for pitch in pitches:
            note_events.append(
                {
                    "pitch_midi": max(21, min(108, pitch)),
                    "start":      chord["start"],
                    "end":        chord["end"] - 0.01,
                    "duration":   chord["end"] - chord["start"] - 0.01,
                    "velocity":   velocity,
                }
            )
    return note_events


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Harmonize a MIDI melody file")
    parser.add_argument("midi",        help="Input MIDI file with melody")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output",    default=None, help="Output chord MIDI")
    parser.add_argument("--tempo",     type=float, default=120.0)
    args = parser.parse_args()

    # Load note events from MIDI
    from utils.midi_utils import load_midi, midi_to_note_events
    pm    = load_midi(args.midi)
    notes = midi_to_note_events(pm)
    notes = [n for n in notes if not n["is_drum"]]
    print(f"Loaded {len(notes)} notes from {args.midi}")

    m2c    = MelodyToChords(checkpoint=args.checkpoint)
    chords = m2c.harmonize(notes, tempo=args.tempo)

    print_chords(chords)

    if args.output:
        from utils.midi_utils import chord_progression_to_midi, save_midi
        pm_out = chord_progression_to_midi(chords, tempo=args.tempo)
        save_midi(pm_out, args.output)
        print(f"\nSaved chord MIDI → {args.output}")
