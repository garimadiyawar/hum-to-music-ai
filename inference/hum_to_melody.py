"""
inference/hum_to_melody.py
───────────────────────────
Stage 1 of the inference pipeline:

    raw humming audio  →  MIDI note event list

Two strategies are available:
  A) Neural:    MelodyTranscriber model (requires trained checkpoint)
  B) Signal:    pitch-detection + note segmentation (always available)

The pipeline selects strategy A if a checkpoint exists, falls back to B.

Public API
----------
    from inference.hum_to_melody import HumToMelody

    h2m = HumToMelody(checkpoint="checkpoints/transcription/best.pt")
    notes = h2m.transcribe("my_hum.wav")   # → List[dict]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.humming_preprocessing import HummingPreprocessor, PreprocessConfig, HummingFeatures
from models.melody_transcriber import MelodyTranscriber, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from utils.audio_utils import load_audio, estimate_tempo, SR_DEFAULT
from utils.pitch_detection import (
    detect_pitch,
    pitch_contour_to_notes,
    hz_to_midi,
)
from utils.music_theory import PITCH_CLASSES


# ─── HumToMelody ─────────────────────────────────────────────────────────────

class HumToMelody:
    """
    Converts humming audio into a sequence of MIDI note events.

    Parameters
    ----------
    checkpoint   : path to MelodyTranscriber checkpoint (optional)
    device       : "cuda" | "cpu" | "mps" | "auto"
    pitch_method : fallback pitch-detection method ("pyin" | "yin" | "crepe")
    beam_size    : beam-search width (1 = greedy)
    config       : PreprocessConfig override
    """

    def __init__(
        self,
        checkpoint:   Optional[str] = None,
        device:       str = "auto",
        pitch_method: str = "pyin",
        beam_size:    int = 1,
        config:       Optional[PreprocessConfig] = None,
    ):
        # ── Device ──────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device(
                "cuda"  if torch.cuda.is_available()         else
                "mps"   if torch.backends.mps.is_available() else
                "cpu"
            )
        else:
            self.device = torch.device(device)

        self.pitch_method = pitch_method
        self.beam_size    = beam_size
        self.preprocessor = HummingPreprocessor(config or PreprocessConfig())
        self.model: Optional[MelodyTranscriber] = None

        # ── Load model (if checkpoint available) ──────────────────────────
        if checkpoint and Path(checkpoint).exists():
            self._load_model(checkpoint)
        else:
            if checkpoint:
                print(f"[HumToMelody] Checkpoint not found: {checkpoint}")
            print("[HumToMelody] Using signal-processing pipeline (no model).")

    def _load_model(self, checkpoint: str):
        """Load a MelodyTranscriber from checkpoint."""
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg  = ckpt.get("model_config", {})
        self.model = MelodyTranscriber(
            n_mels=cfg.get("n_mels", 128),
            d_model=cfg.get("d_model", 512),
            nhead=cfg.get("nhead", 8),
            num_encoder_layers=cfg.get("num_encoder_layers", 6),
            num_decoder_layers=cfg.get("num_decoder_layers", 6),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(f"[HumToMelody] Loaded MelodyTranscriber from {checkpoint}")

    # ── Public API ───────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        sr:    Optional[int] = None,
    ) -> List[Dict]:
        """
        Transcribe humming audio to MIDI note events.

        Parameters
        ----------
        audio : file path, bytes, or waveform ndarray
        sr    : sample rate (required if audio is ndarray)

        Returns
        -------
        List of note event dicts:
            {"pitch_midi": int, "start": float, "end": float,
             "duration": float, "confidence": float, "pitch_hz": float}
        """
        # Pre-process
        features: HummingFeatures = self.preprocessor.process(audio, sr)

        if self.model is not None:
            notes = self._transcribe_neural(features)
        else:
            notes = self._transcribe_signal(features)

        # Post-process: clamp MIDI range, filter very short notes
        notes = _postprocess_notes(notes, min_duration=0.05)
        return notes

    def transcribe_with_features(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        sr:    Optional[int] = None,
    ) -> tuple[List[Dict], HummingFeatures]:
        """Transcribe and also return the full HummingFeatures for downstream use."""
        features = self.preprocessor.process(audio, sr)
        if self.model is not None:
            notes = self._transcribe_neural(features)
        else:
            notes = self._transcribe_signal(features)
        notes = _postprocess_notes(notes, min_duration=0.05)
        return notes, features

    # ── Internal strategies ──────────────────────────────────────────────────

    def _transcribe_neural(self, features: HummingFeatures) -> List[Dict]:
        """
        Use the trained MelodyTranscriber.
        Converts mel spectrogram → token sequence → note events.
        """
        mel_tensor = features.to_tensor(device=self.device).unsqueeze(0)  # (1, 1, n_mels, T)

        if self.beam_size > 1:
            tokens = self.model.beam_search(mel_tensor, beam_size=self.beam_size, max_len=256)
        else:
            seqs   = self.model.greedy_decode(mel_tensor, max_len=256)
            tokens = seqs[0] if seqs else []

        # Convert tokens to note events using timing from pitch contour
        notes = _tokens_to_timed_notes(
            tokens,
            reference_times=features.times,
            reference_f0=features.f0_hz,
            tempo=features.tempo_bpm,
        )
        return notes

    def _transcribe_signal(self, features: HummingFeatures) -> List[Dict]:
        """Use pre-computed note events from pitch-detection pipeline."""
        return features.note_events


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _postprocess_notes(notes: List[Dict], min_duration: float = 0.05) -> List[Dict]:
    """Filter short notes, clamp pitch range, sort by onset."""
    result = []
    for n in notes:
        dur = n.get("duration", n.get("end", 0) - n.get("start", 0))
        if dur < min_duration:
            continue
        pitch = max(21, min(108, int(n["pitch_midi"])))
        result.append({**n, "pitch_midi": pitch})
    result.sort(key=lambda x: x["start"])
    return result


def _tokens_to_timed_notes(
    tokens: List[int],
    reference_times: np.ndarray,
    reference_f0:    np.ndarray,
    tempo:           float = 120.0,
    min_dur:         float = 0.05,
) -> List[Dict]:
    """
    Assign timing to a sequence of predicted MIDI tokens using the reference
    pitch contour as a guide.

    Strategy:
    - Find voiced segments in reference f0.
    - Assign each token to the next voiced segment.
    - Infer duration from segment length.
    """
    if not tokens:
        return []

    sec_per_beat = 60.0 / max(tempo, 1.0)

    # Build voiced segments from reference
    voiced_segments: List[Dict] = []
    in_seg = False
    seg_start = 0.0
    seg_pitches: List[float] = []

    for t, hz in zip(reference_times, reference_f0):
        if hz > 0:
            if not in_seg:
                in_seg = True
                seg_start = float(t)
                seg_pitches = []
            seg_pitches.append(hz)
        else:
            if in_seg:
                avg_hz  = float(np.mean(seg_pitches))
                avg_midi = float(hz_to_midi(np.array([avg_hz]))[0])
                voiced_segments.append({
                    "start": seg_start,
                    "end":   float(t),
                    "pitch_hz":   avg_hz,
                    "pitch_midi": int(round(avg_midi)),
                })
                in_seg = False
                seg_pitches = []

    # If we have voiced segments, map tokens to them
    if voiced_segments:
        notes = []
        for i, tok in enumerate(tokens):
            if i < len(voiced_segments):
                seg = voiced_segments[i]
                notes.append({
                    "pitch_midi": tok,                     # use model's predicted pitch
                    "pitch_hz":   seg["pitch_hz"],
                    "start":      seg["start"],
                    "end":        seg["end"],
                    "duration":   seg["end"] - seg["start"],
                    "confidence": 1.0,
                })
            else:
                # Extra tokens beyond voiced segments: append at end
                if notes:
                    last_end = notes[-1]["end"]
                    dur = sec_per_beat / 2
                    notes.append({
                        "pitch_midi": tok,
                        "pitch_hz":   440.0 * (2 ** ((tok - 69) / 12)),
                        "start":      last_end,
                        "end":        last_end + dur,
                        "duration":   dur,
                        "confidence": 0.8,
                    })
        return notes

    # Fallback: equal-duration notes
    beat_dur = sec_per_beat / 2
    notes = []
    t = 0.0
    for tok in tokens:
        notes.append({
            "pitch_midi": tok,
            "pitch_hz":   440.0 * (2 ** ((tok - 69) / 12)),
            "start":      t,
            "end":        t + beat_dur,
            "duration":   beat_dur,
            "confidence": 0.8,
        })
        t += beat_dur
    return notes


def print_notes(notes: List[Dict], max_show: int = 20):
    """Pretty-print a note event list."""
    print(f"{'#':>3}  {'MIDI':>4}  {'Note':>4}  {'Start':>6}  {'End':>6}  {'Dur':>5}")
    print("─" * 42)
    for i, n in enumerate(notes[:max_show]):
        pc   = PITCH_CLASSES[n["pitch_midi"] % 12]
        octv = (n["pitch_midi"] // 12) - 1
        print(
            f"{i:3d}  {n['pitch_midi']:4d}  {pc+str(octv):>4}  "
            f"{n['start']:6.2f}  {n['end']:6.2f}  {n['duration']:5.2f}"
        )
    if len(notes) > max_show:
        print(f"  … and {len(notes) - max_show} more notes")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe humming audio to MIDI notes")
    parser.add_argument("audio",       help="Path to humming WAV file")
    parser.add_argument("--checkpoint", default=None, help="MelodyTranscriber checkpoint")
    parser.add_argument("--output",    default=None, help="Output MIDI file (.mid)")
    parser.add_argument("--method",    default="pyin", choices=["pyin","yin","crepe"])
    parser.add_argument("--beam",      type=int, default=1)
    args = parser.parse_args()

    print(f"Transcribing: {args.audio}")
    h2m = HumToMelody(
        checkpoint=args.checkpoint,
        pitch_method=args.method,
        beam_size=args.beam,
    )
    notes = h2m.transcribe(args.audio)
    print(f"\nDetected {len(notes)} notes:\n")
    print_notes(notes)

    if args.output:
        from utils.midi_utils import notes_to_midi, save_midi
        pm = notes_to_midi(notes, tempo=120)
        save_midi(pm, args.output)
        print(f"\nSaved MIDI → {args.output}")
