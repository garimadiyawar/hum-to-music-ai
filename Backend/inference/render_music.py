"""
inference/render_music.py
──────────────────────────
Stage 4 (final) of the inference pipeline:

    multi-track MIDI  →  WAV audio

Uses FluidSynth + a SoundFont (.sf2) for high-quality synthesis.
Falls back to a pure-Python sine-wave renderer when FluidSynth is
not available (useful for CI / environments without native libs).

Public API
----------
    from inference.render_music import MusicRenderer

    renderer = MusicRenderer(soundfont="assets/GeneralUser_GS.sf2")
    wav_path = renderer.render(pm, output_path="outputs/song.wav")
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.audio_utils import save_audio, SR_DEFAULT
from utils.midi_utils import (
    arrangement_to_midi,
    save_midi,
    notes_to_midi,
    chord_progression_to_midi,
    render_midi_to_audio,
)


# ─── MusicRenderer ────────────────────────────────────────────────────────────

class MusicRenderer:
    """
    Renders a PrettyMIDI object (or arrangement tracks dict) to a WAV file.

    Parameters
    ----------
    soundfont : path to a .sf2 SoundFont file
                Falls back to ENV var SF2_PATH, then pure-Python renderer.
    sample_rate : output sample rate (default 44100)
    """

    def __init__(
        self,
        soundfont:   Optional[str] = None,
        sample_rate: int = 44100,
    ):
        self.sr = sample_rate
        self.soundfont = soundfont or os.environ.get("SF2_PATH")

        if self.soundfont and not Path(self.soundfont).exists():
            warnings.warn(
                f"SoundFont not found at {self.soundfont}. "
                "Will use pure-Python sine renderer as fallback.",
                RuntimeWarning,
            )
            self.soundfont = None

        if self.soundfont:
            print(f"[MusicRenderer] Using SoundFont: {self.soundfont}")
        else:
            print("[MusicRenderer] Using pure-Python sine renderer (no SoundFont).")

    # ── Public API ───────────────────────────────────────────────────────────

    def render(
        self,
        midi: Union[pretty_midi.PrettyMIDI, Dict[str, List[Dict]]],
        output_path: Union[str, Path],
        tempo: float = 120.0,
        normalize: bool = True,
        fade_out_sec: float = 2.0,
    ) -> Path:
        """
        Render MIDI (or arrangement tracks dict) to WAV.

        Parameters
        ----------
        midi         : PrettyMIDI object OR dict of track_name → note events
        output_path  : where to save the WAV
        tempo        : used only when midi is a tracks dict
        normalize    : peak-normalize output to [-1, 1]
        fade_out_sec : fade-out length at the end (seconds)

        Returns
        -------
        Path to the written WAV file
        """
        # Convert tracks dict → PrettyMIDI if necessary
        if isinstance(midi, dict):
            pm = arrangement_to_midi(midi, tempo=tempo)
        else:
            pm = midi

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Try FluidSynth first, then fall back
        if self.soundfont:
            audio = self._render_fluidsynth(pm)
        else:
            audio = self._render_sine(pm)

        # Mix stereo → mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Fade out
        if fade_out_sec > 0 and len(audio) > 0:
            audio = _apply_fade_out(audio, self.sr, fade_out_sec)

        # Normalize
        if normalize:
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.95

        save_audio(audio, output_path, sr=self.sr)
        print(f"[MusicRenderer] Saved WAV → {output_path}  ({len(audio)/self.sr:.1f}s)")
        return output_path

    def render_from_parts(
        self,
        melody_notes: List[Dict],
        chords:       List[Dict],
        arr_tracks:   Dict[str, List[Dict]],
        output_path:  Union[str, Path],
        tempo:        float = 120.0,
        **render_kwargs,
    ) -> Path:
        """
        Convenience method: combine melody + chords + arrangement tracks
        into a single PrettyMIDI and render.
        """
        import pretty_midi as pm_lib
        from utils.midi_utils import (
            create_midi, add_instrument, add_note, GM_PROGRAMS,
            ARRANGEMENT_TRACKS,
        )

        pm = create_midi(tempo=tempo)

        # Melody
        mel_inst = add_instrument(pm, program=GM_PROGRAMS["acoustic_grand_piano"],
                                   name="Melody")
        for n in melody_notes:
            add_note(mel_inst, n["pitch_midi"], n["start"], n["end"],
                     velocity=n.get("velocity", 90))

        # Chord pad (piano voicing)
        from inference.melody_to_chords import chords_to_note_events
        chord_notes = chords_to_note_events(chords, velocity=60)
        chord_inst  = add_instrument(pm, program=GM_PROGRAMS["electric_piano"], name="Chords")
        for n in chord_notes:
            add_note(chord_inst, n["pitch_midi"], n["start"], n["end"],
                     velocity=n.get("velocity", 60))

        # Arrangement tracks
        for track_name, events in arr_tracks.items():
            if track_name == "melody":
                continue          # already added above
            cfg = ARRANGEMENT_TRACKS.get(track_name, {"program": 0, "is_drum": False})
            inst = add_instrument(pm, program=cfg["program"],
                                   is_drum=cfg.get("is_drum", False),
                                   name=track_name.capitalize())
            for n in events:
                add_note(inst, n["pitch_midi"], n["start"], n["end"],
                         velocity=n.get("velocity", 75))

        return self.render(pm, output_path, tempo=tempo, **render_kwargs)

    # ── Rendering backends ────────────────────────────────────────────────────

    def _render_fluidsynth(self, pm: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Render using FluidSynth via pretty_midi."""
        try:
            audio = pm.fluidsynth(fs=self.sr, sf2_path=self.soundfont)
            return audio.astype(np.float32)
        except Exception as exc:
            warnings.warn(f"FluidSynth failed ({exc}). Falling back to sine renderer.")
            return self._render_sine(pm)

    def _render_sine(self, pm: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Pure-Python fallback renderer using additive sine synthesis.
        Quality is low but always available.
        """
        sr = self.sr
        duration = pm.get_end_time()
        if duration <= 0:
            return np.zeros(sr, dtype=np.float32)

        total_samples = int(duration * sr) + sr    # +1 second tail
        audio = np.zeros(total_samples, dtype=np.float32)

        for inst in pm.instruments:
            is_drum = inst.is_drum
            for note in inst.notes:
                start_s = int(note.start * sr)
                end_s   = int(note.end   * sr)
                n_samp  = max(0, end_s - start_s)
                if n_samp == 0 or start_s >= total_samples:
                    continue

                vel_scale = note.velocity / 127.0 * 0.15

                if is_drum:
                    wave = _drum_wave(note.pitch, n_samp, sr) * vel_scale
                else:
                    hz   = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
                    wave = _sine_wave_adsr(hz, n_samp, sr) * vel_scale

                end_idx = min(start_s + len(wave), total_samples)
                audio[start_s:end_idx] += wave[: end_idx - start_s]

        return audio.astype(np.float32)


# ─── DSP helpers ─────────────────────────────────────────────────────────────

def _sine_wave_adsr(
    freq:    float,
    n_samp:  int,
    sr:      int,
    attack:  float = 0.005,
    decay:   float = 0.05,
    sustain: float = 0.7,
    release: float = 0.1,
    harmonics: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate an ADSR-enveloped sine wave with optional harmonics.

    harmonics : list of (frequency_multiplier, amplitude_scale)
                e.g. [(2, 0.4), (3, 0.2)] adds 2nd and 3rd harmonics
    """
    from typing import Tuple   # noqa: F811

    t = np.linspace(0, n_samp / sr, n_samp, dtype=np.float32)

    # Fundamental
    wave = np.sin(2 * np.pi * freq * t)

    # Add harmonics for richer timbre
    if harmonics is None:
        harmonics = [(2.0, 0.35), (3.0, 0.15), (4.0, 0.07)]
    for mult, amp in harmonics:
        wave += amp * np.sin(2 * np.pi * freq * mult * t)

    # Normalize
    wave /= (1 + sum(a for _, a in harmonics))

    # ADSR envelope
    env = np.ones(n_samp, dtype=np.float32)
    att_s = int(attack  * sr)
    dec_s = int(decay   * sr)
    rel_s = int(release * sr)

    if att_s > 0 and att_s <= n_samp:
        env[:att_s] = np.linspace(0, 1, att_s)
    if dec_s > 0 and att_s + dec_s <= n_samp:
        env[att_s: att_s + dec_s] = np.linspace(1, sustain, dec_s)
    if att_s + dec_s < n_samp:
        env[att_s + dec_s :] = sustain
    if rel_s > 0 and n_samp >= rel_s:
        env[-rel_s:] *= np.linspace(1, 0, rel_s)

    return (wave * env).astype(np.float32)


def _drum_wave(pitch: int, n_samp: int, sr: int) -> np.ndarray:
    """Simplified drum synthesis using noise bursts and tone decay."""
    t = np.linspace(0, n_samp / sr, n_samp, dtype=np.float32)

    if pitch == 36:   # Kick
        env  = np.exp(-t * 25)
        freq = 60 * np.exp(-t * 30)
        wave = np.sin(2 * np.pi * np.cumsum(freq) / sr)
        noise = np.random.randn(n_samp).astype(np.float32) * 0.1
        return ((wave * 0.9 + noise * 0.1) * env).astype(np.float32)

    elif pitch == 38:  # Snare
        env  = np.exp(-t * 40)
        tone = np.sin(2 * np.pi * 200 * t) * 0.3
        noise = np.random.randn(n_samp).astype(np.float32) * 0.7
        return ((tone + noise) * env).astype(np.float32)

    elif pitch in (42, 44):  # Hi-hat closed / pedal
        env  = np.exp(-t * 150)
        noise = np.random.randn(n_samp).astype(np.float32)
        return (noise * env * 0.5).astype(np.float32)

    elif pitch == 46:  # Hi-hat open
        env  = np.exp(-t * 30)
        noise = np.random.randn(n_samp).astype(np.float32)
        return (noise * env * 0.4).astype(np.float32)

    else:  # Generic drum hit
        env   = np.exp(-t * 80)
        noise = np.random.randn(n_samp).astype(np.float32)
        return (noise * env * 0.3).astype(np.float32)


def _apply_fade_out(audio: np.ndarray, sr: int, fade_sec: float) -> np.ndarray:
    """Apply a linear fade-out to the last fade_sec seconds."""
    fade_samples = min(int(fade_sec * sr), len(audio))
    if fade_samples <= 0:
        return audio
    fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    audio = audio.copy()
    audio[-fade_samples:] *= fade
    return audio


# ─── Typing shim ─────────────────────────────────────────────────────────────

from typing import Tuple   # noqa: E402


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a MIDI file to WAV")
    parser.add_argument("midi",         help="Input MIDI file (.mid)")
    parser.add_argument("--output",     default="outputs/rendered.wav")
    parser.add_argument("--soundfont",  default=None, help="Path to .sf2 SoundFont")
    parser.add_argument("--sr",         type=int, default=44100)
    args = parser.parse_args()

    pm   = pretty_midi.PrettyMIDI(args.midi)
    renderer = MusicRenderer(soundfont=args.soundfont, sample_rate=args.sr)
    out_path = renderer.render(pm, args.output)
    print(f"Rendered → {out_path}")
