"""
data/dataset_loader.py
──────────────────────
PyTorch Dataset classes for training all three models:
  1. HummingTranscriptionDataset  – audio mel → MIDI melody tokens
  2. HarmonyDataset               – melody tokens → chord tokens
  3. ArrangementDataset           – melody + chords → multi-track MIDI tokens

Also provides DataLoaders and a unified ``build_dataloaders`` factory.

Supported raw datasets
----------------------
  • MAESTRO   – aligned audio + MIDI piano performances
  • Lakh MIDI – large MIDI-only collection
  • MedleyDB  – multi-track with melody annotations (ground truth f0)
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from data.humming_preprocessing import HummingPreprocessor, PreprocessConfig
from data.midi_processing import MidiParser, MidiTokenizer, scan_midi_directory
from utils.music_theory import (
    NO_CHORD_TOKEN,
    chord_vocab_size,
    SOS_TOKEN,
    EOS_TOKEN,
)
from utils.audio_utils import compute_mel_spectrogram, load_audio, SR_DEFAULT, HOP_DEFAULT, N_MELS_DEFAULT


# ─── 1. Transcription Dataset ────────────────────────────────────────────────

class HummingTranscriptionDataset(Dataset):
    """
    Pairs (mel_spectrogram, midi_token_sequence).

    Built from MAESTRO: the audio side is used as the "humming" input,
    and the aligned MIDI provides ground-truth note sequences.

    When real humming data is unavailable this is the standard proxy.
    """

    PAD = 130
    SOS = SOS_TOKEN
    EOS = EOS_TOKEN

    def __init__(
        self,
        audio_paths: List[Path],
        midi_paths:  List[Path],
        max_mel_frames: int = 1024,
        max_notes:      int = 256,
        sr: int = SR_DEFAULT,
        hop_length: int = HOP_DEFAULT,
        n_mels: int = N_MELS_DEFAULT,
        cache_dir: Optional[Path] = None,
        augment: bool = False,
    ):
        assert len(audio_paths) == len(midi_paths), \
            "audio_paths and midi_paths must be aligned"
        self.audio_paths    = audio_paths
        self.midi_paths     = midi_paths
        self.max_mel_frames = max_mel_frames
        self.max_notes      = max_notes
        self.sr             = sr
        self.hop_length     = hop_length
        self.n_mels         = n_mels
        self.cache_dir      = Path(cache_dir) if cache_dir else None
        self.augment        = augment
        self.tokenizer      = MidiTokenizer()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cache_path = self.cache_dir / f"{idx}.pkl" if self.cache_dir else None
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # ── audio → mel ──
        y, sr = load_audio(self.audio_paths[idx], sr=self.sr, mono=True)
        mel = compute_mel_spectrogram(y, sr=self.sr, n_mels=self.n_mels,
                                       hop_length=self.hop_length)
        if self.augment:
            mel = self._augment_mel(mel)

        # Truncate / pad mel
        T = mel.shape[1]
        if T > self.max_mel_frames:
            start = random.randint(0, T - self.max_mel_frames) if self.augment else 0
            mel = mel[:, start : start + self.max_mel_frames]
        else:
            mel = np.pad(mel, ((0, 0), (0, self.max_mel_frames - T)))

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)   # (1, n_mels, T)

        # ── MIDI → token sequence ──
        parser = MidiParser()
        parser.parse(self.midi_paths[idx])
        tokens = self.tokenizer.encode(
            parser.melody_events,
            add_sos=True,
            add_eos=True,
            max_len=self.max_notes,
        )
        # Pad
        tokens += [self.PAD] * max(0, self.max_notes - len(tokens))
        tokens = tokens[: self.max_notes]
        token_tensor = torch.tensor(tokens, dtype=torch.long)

        item = {
            "mel":          mel_tensor,                                    # (1, n_mels, T)
            "mel_length":   torch.tensor(min(T, self.max_mel_frames)),     # scalar
            "tokens":       token_tensor,                                  # (N,)
            "token_length": torch.tensor(sum(1 for t in tokens if t != self.PAD)),
        }

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(item, f)
        return item

    @staticmethod
    def _augment_mel(mel: np.ndarray) -> np.ndarray:
        """Light augmentation: time warp and mild noise."""
        # Pitch shift (shift bins)
        shift = random.randint(-2, 2)
        if shift != 0:
            mel = np.roll(mel, shift, axis=0)
        # Additive noise
        mel = mel + np.random.randn(*mel.shape).astype(np.float32) * 0.01
        return np.clip(mel, 0.0, 1.0)

    @classmethod
    def from_maestro(
        cls,
        maestro_root: Union[str, Path],
        split: str = "train",
        **kwargs,
    ) -> "HummingTranscriptionDataset":
        """
        Build dataset from MAESTRO directory layout:
            maestro_root/
              year/
                *.wav
                *.midi
        """
        root = Path(maestro_root)
        audio_paths, midi_paths = [], []
        for wav in sorted(root.rglob("*.wav")):
            mid = wav.with_suffix(".midi")
            if not mid.exists():
                mid = wav.with_suffix(".mid")
            if mid.exists():
                audio_paths.append(wav)
                midi_paths.append(mid)

        # Simple split by file index
        N = len(audio_paths)
        if split == "train":
            audio_paths, midi_paths = audio_paths[: int(N * 0.9)], midi_paths[: int(N * 0.9)]
        elif split == "val":
            audio_paths, midi_paths = audio_paths[int(N * 0.9):], midi_paths[int(N * 0.9):]

        return cls(audio_paths, midi_paths, **kwargs)


# ─── 2. Harmony Dataset ──────────────────────────────────────────────────────

class HarmonyDataset(Dataset):
    """
    Pairs (melody_token_sequence, chord_token_sequence).
    Built from Lakh MIDI or any MIDI corpus.
    """

    PAD     = MidiTokenizer.PAD_TOKEN   # 130
    NO_CHORD = NO_CHORD_TOKEN

    def __init__(
        self,
        midi_paths: List[Path],
        max_melody_len: int = 64,
        max_chord_len:  int = 64,
        bar_beats:      int = 4,
    ):
        self.midi_paths     = midi_paths
        self.max_melody_len = max_melody_len
        self.max_chord_len  = max_chord_len
        self.bar_beats      = bar_beats
        self.tokenizer      = MidiTokenizer()

        # Pre-build all (melody, chord) pairs
        self._pairs: List[Tuple[List[int], List[int]]] = []
        self._build()

    def _build(self):
        from data.midi_processing import build_melody_chord_pairs
        for path in self.midi_paths:
            try:
                parser = MidiParser()
                parser.parse(path)
                pairs = build_melody_chord_pairs(parser, bar_beats=self.bar_beats)
                for p in pairs:
                    mel_tok  = [SOS_TOKEN] + p["melody"][:self.max_melody_len - 2] + [EOS_TOKEN]
                    chord_tok = [p["chord"]]
                    self._pairs.append((mel_tok, chord_tok))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mel_tok, chord_tok = self._pairs[idx]
        # Pad melody
        mel_tok = mel_tok + [self.PAD] * max(0, self.max_melody_len - len(mel_tok))
        mel_tok = mel_tok[: self.max_melody_len]

        return {
            "melody": torch.tensor(mel_tok, dtype=torch.long),
            "chord":  torch.tensor(chord_tok[0], dtype=torch.long),
        }

    @classmethod
    def from_lakh(
        cls,
        lakh_root: Union[str, Path],
        max_files: Optional[int] = None,
        **kwargs,
    ) -> "HarmonyDataset":
        midi_paths = scan_midi_directory(lakh_root, max_files=max_files)
        return cls(midi_paths, **kwargs)


# ─── 3. Arrangement Dataset ──────────────────────────────────────────────────

class ArrangementDataset(Dataset):
    """
    Triples (melody_tokens, chord_tokens, arrangement_token_sequence).
    The arrangement sequence is a flattened multi-track MIDI-like event sequence.
    """

    PAD = MidiTokenizer.PAD_TOKEN

    def __init__(
        self,
        midi_paths:   List[Path],
        max_src_len:  int = 128,
        max_tgt_len:  int = 512,
    ):
        self.midi_paths  = midi_paths
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer   = MidiTokenizer()
        self._items: List[Dict] = []
        self._build()

    def _build(self):
        from utils.midi_utils import note_events_to_tokens
        for path in self.midi_paths:
            try:
                parser = MidiParser()
                parser.parse(path)

                mel_tokens = self.tokenizer.encode(
                    parser.melody_events, add_sos=True, add_eos=True
                )
                chord_tokens = [c["token"] for c in parser.chord_events]

                # Full arrangement = all non-drum, non-melody notes
                other_notes = [
                    e for e in parser.note_events
                    if not e["is_drum"] and e not in parser.melody_events
                ]
                arr_tokens = note_events_to_tokens(other_notes)

                self._items.append(
                    {
                        "melody": mel_tokens,
                        "chords": chord_tokens,
                        "arrangement": arr_tokens,
                    }
                )
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self._items[idx]

        def pad(seq, length):
            seq = seq[:length]
            return seq + [self.PAD] * max(0, length - len(seq))

        mel  = pad(item["melody"],      self.max_src_len)
        chrd = pad(item["chords"],      self.max_src_len)
        arr  = pad(item["arrangement"], self.max_tgt_len)

        return {
            "melody":      torch.tensor(mel,  dtype=torch.long),
            "chords":      torch.tensor(chrd, dtype=torch.long),
            "arrangement": torch.tensor(arr,  dtype=torch.long),
        }


# ─── DataLoader factory ──────────────────────────────────────────────────────

def build_dataloaders(
    dataset: Dataset,
    val_split: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split a dataset into train / val DataLoaders.
    """
    N = len(dataset)
    n_val = max(1, int(N * val_split))
    n_train = N - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ─── Example ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Demo: build HarmonyDataset from a directory of MIDI files
    midi_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/midi_demo"

    if not Path(midi_dir).exists():
        print(f"Directory {midi_dir} not found. Creating a synthetic MIDI for demo …")
        Path("/tmp/midi_demo").mkdir(exist_ok=True)
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        inst = pretty_midi.Instrument(program=0, name="Piano")
        for i, (pitch, t) in enumerate(zip([60, 64, 67, 72], [0, 0.5, 1.0, 1.5])):
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=t, end=t + 0.4))
        pm.instruments.append(inst)
        pm.write("/tmp/midi_demo/demo.mid")
        midi_dir = "/tmp/midi_demo"

    midi_paths = scan_midi_directory(midi_dir, max_files=10)
    print(f"Found {len(midi_paths)} MIDI files")

    if midi_paths:
        ds = HarmonyDataset(midi_paths, max_melody_len=32)
        print(f"HarmonyDataset: {len(ds)} pairs")
        if len(ds) > 0:
            sample = ds[0]
            print(f"  melody: {sample['melody'][:10]}…")
            print(f"  chord:  {sample['chord'].item()}")
