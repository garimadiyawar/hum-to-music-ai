"""
utils/pitch_detection.py
────────────────────────
Multi-backend pitch detection: CREPE (neural), pYIN (probabilistic),
and YIN (classical).  All return the same canonical format.

Canonical output
----------------
    times      : np.ndarray  shape (T,)   – frame centre times in seconds
    frequencies: np.ndarray  shape (T,)   – estimated f0 in Hz (0 → unvoiced)
    confidence : np.ndarray  shape (T,)   – [0, 1] voicing probability
    voiced_mask: np.ndarray  shape (T,)   – bool, True where voiced
"""

from __future__ import annotations

from typing import Literal, Tuple

import librosa
import numpy as np


# ─── Type aliases ────────────────────────────────────────────────────────────

PitchResult = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def hz_to_midi(hz: np.ndarray, unvoiced_val: float = 0.0) -> np.ndarray:
    """Convert frequency array (Hz) to MIDI note numbers.  0 Hz → unvoiced_val."""
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = np.where(
            hz > 0,
            12.0 * np.log2(hz / 440.0) + 69.0,
            unvoiced_val,
        )
    return midi.astype(np.float32)


def midi_to_hz(midi: np.ndarray, unvoiced_val: float = 0.0) -> np.ndarray:
    """Convert MIDI note numbers to Hz.  Values ≤ 0 → unvoiced_val Hz."""
    return np.where(
        midi > 0,
        440.0 * (2.0 ** ((midi - 69.0) / 12.0)),
        unvoiced_val,
    ).astype(np.float32)


def smooth_pitch(
    frequencies: np.ndarray,
    voiced_mask: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """
    Median-smooth voiced pitch contour.
    Unvoiced frames are left at 0.
    """
    smoothed = frequencies.copy()
    half = window // 2
    for i in range(len(frequencies)):
        if voiced_mask[i]:
            lo, hi = max(0, i - half), min(len(frequencies), i + half + 1)
            voiced_segment = frequencies[lo:hi][voiced_mask[lo:hi]]
            if len(voiced_segment) > 0:
                smoothed[i] = float(np.median(voiced_segment))
    return smoothed


# ─── pYIN backend ────────────────────────────────────────────────────────────

def detect_pyin(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    confidence_threshold: float = 0.5,
) -> PitchResult:
    """
    Probabilistic YIN pitch estimator (librosa.pyin).
    Most robust for noisy humming.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0 = np.where(voiced_flag, f0, 0.0).astype(np.float32)
    confidence = voiced_probs.astype(np.float32)
    voiced_mask = voiced_flag & (confidence >= confidence_threshold)
    f0 = np.where(voiced_mask, f0, 0.0).astype(np.float32)

    return times.astype(np.float32), f0, confidence, voiced_mask


# ─── YIN backend ─────────────────────────────────────────────────────────────

def detect_yin(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    frame_length: int = 2048,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    confidence_threshold: float = 0.5,
) -> PitchResult:
    """
    Deterministic YIN via librosa.yin.
    Faster than pYIN, slightly less robust.
    """
    f0 = librosa.yin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float32)

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length).astype(np.float32)

    # YIN returns fmin for unvoiced frames → treat them as unvoiced
    voiced_mask = f0 > fmin
    confidence = voiced_mask.astype(np.float32)
    f0 = np.where(voiced_mask, f0, 0.0)

    return times, f0, confidence, voiced_mask


# ─── CREPE backend ───────────────────────────────────────────────────────────

def detect_crepe(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    model_capacity: Literal["tiny", "small", "medium", "large", "full"] = "small",
    confidence_threshold: float = 0.5,
    viterbi: bool = True,
) -> PitchResult:
    """
    CREPE neural pitch tracker.
    Requires the ``crepe`` package.

    ``viterbi=True`` applies Viterbi decoding for smoother trajectories.
    """
    try:
        import crepe  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "CREPE not installed.  Run: pip install crepe"
        ) from exc

    hop_size_ms = hop_length / sr  # crepe expects step_size in seconds
    time_, frequency, confidence, activation = crepe.predict(
        y,
        sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        center=True,
        step_size=hop_size_ms * 1000,  # ms
        verbose=0,
    )

    voiced_mask = confidence >= confidence_threshold
    f0 = np.where(voiced_mask, frequency, 0.0).astype(np.float32)
    times = time_.astype(np.float32)
    conf = confidence.astype(np.float32)

    return times, f0, conf, voiced_mask.astype(bool)


# ─── Unified interface ───────────────────────────────────────────────────────

def detect_pitch(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    method: Literal["pyin", "yin", "crepe"] = "pyin",
    confidence_threshold: float = 0.5,
    smooth: bool = True,
    smooth_window: int = 5,
    **kwargs,
) -> PitchResult:
    """
    Unified pitch detection entry point.

    Parameters
    ----------
    y                    : mono audio waveform
    sr                   : sample rate
    hop_length           : analysis hop length in samples
    method               : "pyin" | "yin" | "crepe"
    confidence_threshold : min confidence to be considered voiced
    smooth               : apply median smoothing to pitch contour
    smooth_window        : window length for smoothing

    Returns
    -------
    times, frequencies, confidence, voiced_mask
    """
    dispatch = {
        "pyin":  detect_pyin,
        "yin":   detect_yin,
        "crepe": detect_crepe,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown pitch method '{method}'. Choose from {list(dispatch)}")

    times, f0, conf, voiced = dispatch[method](
        y,
        sr=sr,
        hop_length=hop_length,
        confidence_threshold=confidence_threshold,
        **kwargs,
    )

    if smooth and voiced.any():
        f0 = smooth_pitch(f0, voiced, window=smooth_window)

    return times, f0, conf, voiced


# ─── Pitch contour → note events ─────────────────────────────────────────────

def pitch_contour_to_notes(
    times: np.ndarray,
    f0: np.ndarray,
    voiced_mask: np.ndarray,
    min_note_duration: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
) -> list[dict]:
    """
    Convert a frame-level pitch contour into a list of note events.

    Each note event:
        {
            "pitch_midi" : int,   – rounded MIDI note
            "pitch_hz"   : float, – mean Hz of the note
            "start"      : float, – onset time in seconds
            "end"        : float, – offset time in seconds
            "duration"   : float,
            "confidence" : float, – mean confidence
        }

    Parameters
    ----------
    pitch_tolerance_cents : merge adjacent frames if pitch difference
                            is within this many cents.
    """
    notes: list[dict] = []

    midi_contour = hz_to_midi(f0, unvoiced_val=-1.0)

    in_note = False
    note_start = 0.0
    note_pitches: list[float] = []
    note_confs: list[float] = []
    prev_midi = -1.0

    cents_per_semitone = 100.0

    for i, (t, voiced, midi_f, hz_f) in enumerate(
        zip(times, voiced_mask, midi_contour, f0)
    ):
        if voiced and hz_f > 0:
            cents_diff = abs(midi_f - prev_midi) * cents_per_semitone if prev_midi >= 0 else 0.0
            if not in_note or cents_diff > pitch_tolerance_cents:
                # Save previous note
                if in_note and len(note_pitches) > 0:
                    dur = times[i - 1] - note_start
                    if dur >= min_note_duration:
                        avg_hz = float(np.mean(note_pitches))
                        notes.append(
                            {
                                "pitch_midi": int(round(hz_to_midi(np.array([avg_hz]))[0])),
                                "pitch_hz": avg_hz,
                                "start": note_start,
                                "end": float(times[i - 1]),
                                "duration": dur,
                                "confidence": float(np.mean(note_confs)),
                            }
                        )
                # Start new note
                in_note = True
                note_start = float(t)
                note_pitches = [hz_f]
                note_confs = []
                prev_midi = midi_f
            else:
                note_pitches.append(hz_f)
                prev_midi = float(np.mean(hz_to_midi(np.array(note_pitches))))
        else:
            # Unvoiced → close current note
            if in_note and len(note_pitches) > 0:
                dur = float(t) - note_start
                if dur >= min_note_duration:
                    avg_hz = float(np.mean(note_pitches))
                    notes.append(
                        {
                            "pitch_midi": int(round(hz_to_midi(np.array([avg_hz]))[0])),
                            "pitch_hz": avg_hz,
                            "start": note_start,
                            "end": float(t),
                            "duration": dur,
                            "confidence": float(np.mean(note_confs)) if note_confs else 1.0,
                        }
                    )
            in_note = False
            note_pitches = []
            note_confs = []
            prev_midi = -1.0

    # Close final note
    if in_note and len(note_pitches) > 0:
        dur = float(times[-1]) - note_start
        if dur >= min_note_duration:
            avg_hz = float(np.mean(note_pitches))
            notes.append(
                {
                    "pitch_midi": int(round(hz_to_midi(np.array([avg_hz]))[0])),
                    "pitch_hz": avg_hz,
                    "start": note_start,
                    "end": float(times[-1]),
                    "duration": dur,
                    "confidence": float(np.mean(note_confs)) if note_confs else 1.0,
                }
            )

    return notes


# ─── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from utils.audio_utils import load_audio

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        y, sr = load_audio(path)
    else:
        sr = 22050
        t = np.linspace(0, 3, 3 * sr)
        # Simulate humming: C4 for 1s, E4 for 1s, G4 for 1s
        y = (
            np.sin(2 * np.pi * 261.63 * t[:sr]) * 0.5
            + np.sin(2 * np.pi * 329.63 * t[sr : 2 * sr]) * 0.5
            + np.sin(2 * np.pi * 392.00 * t[2 * sr :]) * 0.5
        ).astype(np.float32)

    times, f0, conf, voiced = detect_pitch(y, sr=sr, method="pyin")
    print(f"Frames: {len(times)}, Voiced: {voiced.sum()}")

    notes = pitch_contour_to_notes(times, f0, voiced)
    for n in notes:
        print(f"  MIDI {n['pitch_midi']:3d}  {n['start']:.2f}s – {n['end']:.2f}s  "
              f"({n['duration']:.2f}s)")
