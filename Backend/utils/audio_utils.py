"""
utils/audio_utils.py
────────────────────
Core audio loading, feature extraction, and pre-processing utilities.
All functions are stateless and safe to call from multiple workers.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch


# ─── Constants ───────────────────────────────────────────────────────────────

SR_DEFAULT = 22050          # canonical sample-rate used throughout the system
HOP_DEFAULT = 256
N_FFT_DEFAULT = 2048
N_MELS_DEFAULT = 128


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_audio(
    path: Union[str, Path, bytes],
    sr: int = SR_DEFAULT,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file to a float32 waveform.

    Parameters
    ----------
    path : file path OR raw bytes
    sr   : target sample rate (resamples if necessary)
    mono : mix down to mono
    normalize : peak-normalize to [-1, 1]

    Returns
    -------
    (waveform, sample_rate)  –  waveform shape: (samples,) for mono
    """
    if isinstance(path, (bytes, bytearray)):
        # Load from in-memory bytes (e.g. uploaded file)
        buf = io.BytesIO(path)
        y, file_sr = sf.read(buf, dtype="float32", always_2d=False)
    else:
        y, file_sr = librosa.load(str(path), sr=None, mono=mono)

    # Resample
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # Mono mix
    if mono and y.ndim > 1:
        y = y.mean(axis=0)

    # Peak normalize
    if normalize:
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak

    return y.astype(np.float32), sr


def save_audio(
    waveform: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    sr: int = SR_DEFAULT,
) -> None:
    """Write a float32 waveform to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    sf.write(str(path), waveform.astype(np.float32), sr)


# ─── Spectral Features ───────────────────────────────────────────────────────

def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = SR_DEFAULT,
    n_mels: int = N_MELS_DEFAULT,
    n_fft: int = N_FFT_DEFAULT,
    hop_length: int = HOP_DEFAULT,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    to_db: bool = True,
    ref: float = 1.0,
) -> np.ndarray:
    """
    Compute log-mel spectrogram.

    Returns
    -------
    mel : ndarray shape (n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    if to_db:
        mel = librosa.power_to_db(mel, ref=ref, top_db=80.0)
        # Normalize to [0, 1]
        mel = (mel + 80.0) / 80.0
    return mel.astype(np.float32)


def compute_mfcc(
    y: np.ndarray,
    sr: int = SR_DEFAULT,
    n_mfcc: int = 40,
    n_fft: int = N_FFT_DEFAULT,
    hop_length: int = HOP_DEFAULT,
) -> np.ndarray:
    """Return MFCC matrix (n_mfcc, frames)."""
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    ).astype(np.float32)


def compute_chroma(
    y: np.ndarray,
    sr: int = SR_DEFAULT,
    hop_length: int = HOP_DEFAULT,
    n_fft: int = N_FFT_DEFAULT,
) -> np.ndarray:
    """Return chroma feature matrix (12, frames)."""
    return librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    ).astype(np.float32)


def compute_rms(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = HOP_DEFAULT,
) -> np.ndarray:
    """RMS energy per frame, shape (frames,)."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    return rms[0].astype(np.float32)


# ─── Tempo / Beat ────────────────────────────────────────────────────────────

def estimate_tempo(
    y: np.ndarray,
    sr: int = SR_DEFAULT,
    hop_length: int = HOP_DEFAULT,
    start_bpm: float = 120.0,
) -> Tuple[float, np.ndarray]:
    """
    Estimate global tempo (BPM) and beat positions.

    Returns
    -------
    (tempo_bpm, beat_times)
    """
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, start_bpm=start_bpm, units="frames"
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return float(tempo), beat_times


# ─── Signal Utilities ────────────────────────────────────────────────────────

def trim_silence(
    y: np.ndarray,
    top_db: float = 30.0,
    sr: int = SR_DEFAULT,
) -> np.ndarray:
    """Trim leading/trailing silence."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return trimmed


def pad_or_truncate(
    y: np.ndarray,
    max_samples: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad with zeros or truncate waveform to exactly max_samples."""
    if len(y) >= max_samples:
        return y[:max_samples]
    pad_width = max_samples - len(y)
    return np.pad(y, (0, pad_width), constant_values=pad_value)


def frames_to_time(frames: np.ndarray, sr: int = SR_DEFAULT, hop: int = HOP_DEFAULT) -> np.ndarray:
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def time_to_frames(times: np.ndarray, sr: int = SR_DEFAULT, hop: int = HOP_DEFAULT) -> np.ndarray:
    return librosa.time_to_frames(times, sr=sr, hop_length=hop)


# ─── Torch Tensor Helpers ────────────────────────────────────────────────────

def mel_to_tensor(mel: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert (n_mels, T) numpy mel to (1, n_mels, T) float tensor.
    Suitable for feeding directly into CNN.
    """
    t = torch.from_numpy(mel).unsqueeze(0)          # (1, n_mels, T)
    if device is not None:
        t = t.to(device)
    return t


def batch_mel(mels: list[np.ndarray], pad_value: float = 0.0) -> torch.Tensor:
    """
    Collate a list of (n_mels, T_i) arrays → (B, 1, n_mels, T_max) tensor.
    """
    max_t = max(m.shape[1] for m in mels)
    n_mels = mels[0].shape[0]
    out = np.full((len(mels), n_mels, max_t), pad_value, dtype=np.float32)
    for i, m in enumerate(mels):
        out[i, :, : m.shape[1]] = m
    return torch.from_numpy(out).unsqueeze(1)       # (B, 1, n_mels, T_max)


# ─── Example Usage ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        # Generate synthetic sine wave for demo
        sr = SR_DEFAULT
        t = np.linspace(0, 2, 2 * sr)
        y = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        print(f"Demo sine wave: {y.shape}, sr={sr}")
    else:
        y, sr = load_audio(path)
        print(f"Loaded: {y.shape}, sr={sr}")

    mel = compute_mel_spectrogram(y, sr)
    print(f"Mel spectrogram: {mel.shape}")

    tempo, beats = estimate_tempo(y, sr)
    print(f"Estimated tempo: {tempo:.1f} BPM, {len(beats)} beats")

    tensor = mel_to_tensor(mel)
    print(f"Tensor shape: {tensor.shape}")
