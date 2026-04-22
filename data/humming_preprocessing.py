"""
data/humming_preprocessing.py
──────────────────────────────
Pre-processing pipeline that transforms raw humming audio into the
feature representations expected by the transcription model.

Pipeline
--------
raw WAV → trim silence → normalize → mel spectrogram
                                   → pitch contour
                                   → note events (symbolic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from utils.audio_utils import (
    load_audio,
    compute_mel_spectrogram,
    compute_chroma,
    estimate_tempo,
    trim_silence,
    pad_or_truncate,
    mel_to_tensor,
    SR_DEFAULT,
    HOP_DEFAULT,
    N_FFT_DEFAULT,
    N_MELS_DEFAULT,
)
from utils.pitch_detection import (
    detect_pitch,
    pitch_contour_to_notes,
    hz_to_midi,
)
from utils.music_theory import (
    key_from_notes,
    PITCH_CLASSES,
)


# ─── Pre-processing config ───────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    sr: int = SR_DEFAULT
    hop_length: int = HOP_DEFAULT
    n_fft: int = N_FFT_DEFAULT
    n_mels: int = N_MELS_DEFAULT
    fmin: float = 50.0
    fmax: float = 2000.0
    max_duration: float = 30.0          # seconds; longer clips are cropped
    min_note_duration: float = 0.05     # seconds
    pitch_method: str = "pyin"          # "pyin" | "yin" | "crepe"
    confidence_threshold: float = 0.55
    pitch_smooth: bool = True
    pitch_smooth_window: int = 5
    top_db_trim: float = 35.0           # silence trim threshold (dB)
    normalize: bool = True


# ─── Processed output dataclass ──────────────────────────────────────────────

@dataclass
class HummingFeatures:
    """Container for all features extracted from one humming clip."""
    # Waveform
    waveform: np.ndarray                 # (T,) float32

    # Spectral
    mel: np.ndarray                      # (n_mels, frames) float32 in [0,1]
    chroma: np.ndarray                   # (12, frames) float32

    # Pitch contour
    times: np.ndarray                    # (frames,) seconds
    f0_hz: np.ndarray                    # (frames,) Hz, 0 = unvoiced
    f0_midi: np.ndarray                  # (frames,) MIDI, 0 = unvoiced
    confidence: np.ndarray               # (frames,) [0,1]
    voiced_mask: np.ndarray              # (frames,) bool

    # Symbolic
    note_events: List[Dict]              # pitch/start/end/duration/confidence
    tempo_bpm: float
    key_root_pc: int                     # 0-11
    key_scale: str                       # "major" / "natural_minor"

    # Meta
    sample_rate: int = SR_DEFAULT
    duration: float = 0.0

    def to_tensor(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return mel as (1, n_mels, T) tensor, ready for the CNN encoder."""
        return mel_to_tensor(self.mel, device=device)

    def midi_sequence(self) -> List[int]:
        """Return list of MIDI pitch values from note events (for seq2seq models)."""
        return [ev["pitch_midi"] for ev in self.note_events]


# ─── Main preprocessor class ─────────────────────────────────────────────────

class HummingPreprocessor:
    """
    End-to-end preprocessor for humming audio.

    Usage
    -----
    >>> preprocessor = HummingPreprocessor()
    >>> features = preprocessor.process("my_humming.wav")
    >>> mel_tensor = features.to_tensor()
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()

    # ── public API ──────────────────────────────────────────────────────────

    def process(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        sr: Optional[int] = None,
    ) -> HummingFeatures:
        """
        Full preprocessing pipeline.

        Parameters
        ----------
        audio : file path, raw bytes, or already-loaded waveform
        sr    : sample rate if audio is a waveform ndarray

        Returns
        -------
        HummingFeatures dataclass
        """
        y, sr = self._load(audio, sr)
        y = self._preprocess_waveform(y, sr)
        mel = self._mel_spectrogram(y, sr)
        chroma = compute_chroma(y, sr=sr, hop_length=self.cfg.hop_length, n_fft=self.cfg.n_fft)
        times, f0, conf, voiced = self._detect_pitch(y, sr)
        f0_midi = hz_to_midi(f0)
        notes = pitch_contour_to_notes(
            times, f0, voiced,
            min_note_duration=self.cfg.min_note_duration,
        )

        tempo, _ = estimate_tempo(y, sr=sr, hop_length=self.cfg.hop_length)
        # fallback if tempo detection fails
        if tempo is None or tempo <= 0:
            tempo = 90.0

        midi_notes = [n["pitch_midi"] for n in notes]
        key_root, key_scale = key_from_notes(midi_notes) if midi_notes else (0, "major")
        melody_end = max(n["end"] for n in notes) if notes else float(len(y)) / sr
        
        return HummingFeatures(
            waveform=y,
            mel=mel,
            chroma=chroma,
            times=times,
            f0_hz=f0,
            f0_midi=f0_midi,
            confidence=conf,
            voiced_mask=voiced,
            note_events=notes,
            tempo_bpm=float(tempo),
            key_root_pc=key_root,
            key_scale=key_scale,
            sample_rate=sr,
            duration=melody_end,
        )

    def process_batch(
        self,
        paths: List[Union[str, Path]],
    ) -> List[HummingFeatures]:
        """Process a list of audio files sequentially."""
        return [self.process(p) for p in paths]

    # ── internal steps ───────────────────────────────────────────────────────

    def _load(
        self,
        audio: Union[str, Path, bytes, np.ndarray],
        sr: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        """Load and resample to canonical sample rate."""
        if isinstance(audio, np.ndarray):
            target_sr = sr or self.cfg.sr
            if sr is not None and sr != self.cfg.sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.cfg.sr)
            return audio.astype(np.float32), self.cfg.sr
        return load_audio(audio, sr=self.cfg.sr, mono=True, normalize=self.cfg.normalize)

    def _preprocess_waveform(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Trim silence, crop to max duration, re-normalize."""
        y = trim_silence(y, top_db=self.cfg.top_db_trim, sr=sr)
        max_samples = int(self.cfg.max_duration * sr)
        y = pad_or_truncate(y, max_samples)
        if self.cfg.normalize:
            peak = np.abs(y).max()
            if peak > 0:
                y = y / peak
        return y

    def _mel_spectrogram(self, y: np.ndarray, sr: int) -> np.ndarray:
        return compute_mel_spectrogram(
            y,
            sr=sr,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            fmin=self.cfg.fmin,
            fmax=self.cfg.fmax,
            to_db=True,
        )

    def _detect_pitch(
        self, y: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return detect_pitch(
            y,
            sr=sr,
            hop_length=self.cfg.hop_length,
            method=self.cfg.pitch_method,
            confidence_threshold=self.cfg.confidence_threshold,
            smooth=self.cfg.pitch_smooth,
            smooth_window=self.cfg.pitch_smooth_window,
        )


# ─── Collation for DataLoader ─────────────────────────────────────────────────

def collate_humming_batch(
    batch: List[HummingFeatures],
    max_mel_frames: Optional[int] = None,
    max_note_events: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of HummingFeatures into padded tensors.

    Returns
    -------
    dict with keys:
        "mel"         : (B, 1, n_mels, T_mel)
        "mel_lengths" : (B,)  – actual frame counts
        "f0_midi"     : (B, T_f0)
        "note_midi"   : (B, N)  – padded MIDI note sequences
        "note_lengths": (B,)
    """
    B = len(batch)
    n_mels = batch[0].mel.shape[0]

    # Determine padding lengths
    mel_lens = [b.mel.shape[1] for b in batch]
    T_mel = max_mel_frames or max(mel_lens)

    note_lens = [len(b.note_events) for b in batch]
    N = max_note_events or max(note_lens)

    f0_lens = [len(b.f0_midi) for b in batch]
    T_f0 = max(f0_lens)

    mel_batch = np.zeros((B, 1, n_mels, T_mel), dtype=np.float32)
    f0_batch  = np.zeros((B, T_f0), dtype=np.float32)
    note_batch = np.full((B, N), fill_value=-1, dtype=np.int64)  # -1 = PAD

    for i, feat in enumerate(batch):
        t = min(feat.mel.shape[1], T_mel)
        mel_batch[i, 0, :, :t] = feat.mel[:, :t]

        tf = min(len(feat.f0_midi), T_f0)
        f0_batch[i, :tf] = feat.f0_midi[:tf]

        n = min(len(feat.note_events), N)
        for j, ev in enumerate(feat.note_events[:n]):
            note_batch[i, j] = ev["pitch_midi"]

    return {
        "mel":          torch.from_numpy(mel_batch),
        "mel_lengths":  torch.tensor(mel_lens, dtype=torch.long),
        "f0_midi":      torch.from_numpy(f0_batch),
        "note_midi":    torch.from_numpy(note_batch),
        "note_lengths": torch.tensor(note_lens, dtype=torch.long),
    }


# ─── Example / smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    preprocessor = HummingPreprocessor()

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Processing: {path}")
        features = preprocessor.process(path)
    else:
        # Synthetic humming (C-E-G glissando)
        sr = SR_DEFAULT
        dur = 3.0
        t = np.linspace(0, dur, int(dur * sr))
        # Frequency glide: 261 Hz → 392 Hz
        freq = np.linspace(261.63, 392.0, len(t))
        y = (np.sin(2 * np.pi * np.cumsum(freq) / sr) * 0.6).astype(np.float32)
        print("Processing synthetic humming …")
        features = preprocessor.process(y, sr=sr)

    print(f"\n── HummingFeatures ─────────────────────────────")
    print(f"  Duration      : {features.duration:.2f}s")
    print(f"  Mel shape     : {features.mel.shape}")
    print(f"  Tempo         : {features.tempo_bpm:.1f} BPM")
    print(f"  Key           : {PITCH_CLASSES[features.key_root_pc]} {features.key_scale}")
    print(f"  Voiced frames : {features.voiced_mask.sum()} / {len(features.voiced_mask)}")
    print(f"  Note events   : {len(features.note_events)}")
    for n in features.note_events[:8]:
        print(f"    MIDI {n['pitch_midi']:3d}  {n['start']:.2f}–{n['end']:.2f}s")
    if len(features.note_events) > 8:
        print(f"    … and {len(features.note_events) - 8} more")
