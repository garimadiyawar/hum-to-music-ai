#!/usr/bin/env python3
"""
main.py
───────
Entry point for the Hum-to-Music AI system.

Full pipeline
─────────────
  1. Load + preprocess humming audio
  2. Transcribe melody (neural or signal-processing)
  3. Detect key and tempo
  4. Generate chord progression
  5. Generate multi-track arrangement
  6. (Optional) Expand short melody into full song structure
  7. Export MIDI
  8. Render to WAV

Usage examples
──────────────
  # Basic (all rule-based, no checkpoints needed):
  python main.py my_hum.wav

  # With trained models:
  python main.py my_hum.wav \
      --transcription_ckpt checkpoints/transcription/best.pt \
      --harmony_ckpt       checkpoints/harmony/best.pt \
      --arrangement_ckpt   checkpoints/arrangement/best.pt \
      --soundfont          assets/GeneralUser_GS.sf2

  # Custom output directory, specific BPM, full song expansion:
  python main.py my_hum.wav --output_dir results/ --tempo 140 --expand_song

  # Training modes (trains each model sequentially):
  python main.py --train transcription --maestro_root /data/maestro
  python main.py --train harmony       --lakh_root    /data/lakh
  python main.py --train arrangement   --lakh_root    /data/lakh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Project imports ──────────────────────────────────────────────────────────
from data.humming_preprocessing import HummingPreprocessor, PreprocessConfig
from inference.hum_to_melody import HumToMelody, print_notes
from inference.melody_to_chords import MelodyToChords, print_chords
from inference.arrangement_generator import ArrangementGenerator, print_arrangement
from inference.render_music import MusicRenderer
from models.composition_transformer import RuleBasedComposer, CompositionConfig
from utils.audio_utils import SR_DEFAULT, save_audio
from utils.midi_utils import (
    arrangement_to_midi,
    notes_to_midi,
    chord_progression_to_midi,
    save_midi,
)
from utils.music_theory import PITCH_CLASSES


# ─── Banner ───────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║          Hum-to-Music AI System  🎵                  ║
║  Hum → Melody → Harmony → Arrangement → WAV         ║
╚══════════════════════════════════════════════════════╝
"""


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(args: argparse.Namespace) -> dict:
    """
    Execute the full hum-to-music pipeline.

    Returns
    -------
    dict with keys: "notes", "chords", "tracks", "midi_path", "wav_path"
    """
    print(BANNER)
    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Preprocess audio ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 │ Audio preprocessing …")
    print("=" * 60)
    preprocessor = HummingPreprocessor(
        PreprocessConfig(pitch_method=args.pitch_method)
    )
    features = preprocessor.process(args.audio)
    print(f"  Duration  : {features.duration:.2f}s")
    print(f"  Tempo est : {features.tempo_bpm:.1f} BPM")
    print(f"  Key est   : {PITCH_CLASSES[features.key_root_pc]} {features.key_scale}")
    print(f"  Mel shape : {features.mel.shape}")

    tempo = args.tempo if args.tempo else features.tempo_bpm

    # ── Step 2: Melody transcription ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 │ Melody transcription …")
    print("=" * 60)
    h2m = HumToMelody(
        checkpoint=args.transcription_ckpt,
        pitch_method=args.pitch_method,
        beam_size=args.beam_size,
    )
    notes = h2m.transcribe(args.audio)
    print(f"  Detected {len(notes)} notes")
    if args.verbose:
        print_notes(notes, max_show=16)

    if not notes:
        print("  WARNING: No notes detected. Check audio input.")
        return {}

    # ── Step 3 (optional): Song structure expansion ───────────────────────────
    if args.expand_song:
        print("\n" + "=" * 60)
        print("STEP 3 │ Expanding melody into song structure …")
        print("=" * 60)
        comp_cfg = CompositionConfig(
            structure_template=args.song_style,
            default_tempo=tempo,
        )
        composer = RuleBasedComposer(comp_cfg)
        notes, section_labels = composer.compose(
            notes,
            key_root_pc=features.key_root_pc,
            key_scale=features.key_scale,
        )
        print(f"  Expanded to {len(notes)} notes across {len(set(section_labels))} sections")
        sections = {}
        for note, label in zip(notes, section_labels):
            sections.setdefault(label, 0)
            sections[label] += 1
        for sec, cnt in sections.items():
            print(f"    {sec}: {cnt} notes")

    # ── Step 4: Harmony generation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 │ Generating chord progression …")
    print("=" * 60)
    m2c = MelodyToChords(
        checkpoint=args.harmony_ckpt,
        temperature=args.temperature,
    )
    duration = max(n["end"] for n in notes) if notes else 8.0
    chords   = m2c.harmonize(notes, duration=duration, tempo=tempo)
    print(f"  Generated {len(chords)} chords")
    if args.verbose:
        print_chords(chords)

    # ── Step 5: Arrangement ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 │ Generating multi-track arrangement …")
    print("=" * 60)
    arr_gen = ArrangementGenerator(
        checkpoint=args.arrangement_ckpt,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    tracks = arr_gen.arrange(
        melody_notes=notes,
        chords=chords,
        tempo=tempo,
        duration=duration,
        key_root_pc=features.key_root_pc,
        key_scale=features.key_scale,
    )
    print_arrangement(tracks)

    # ── Step 6: Export MIDI ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6 │ Exporting MIDI …")
    print("=" * 60)

    # Melody MIDI
    mel_midi_path = output_dir / "melody.mid"
    pm_mel = notes_to_midi(notes, tempo=tempo, instrument_name="Melody")
    save_midi(pm_mel, mel_midi_path)

    # Chord MIDI
    chord_midi_path = output_dir / "chords.mid"
    pm_chord = chord_progression_to_midi(chords, tempo=tempo)
    save_midi(pm_chord, chord_midi_path)

    # Full arrangement MIDI
    arr_midi_path = output_dir / "arrangement.mid"
    pm_arr = arrangement_to_midi(tracks, tempo=tempo)
    save_midi(pm_arr, arr_midi_path)
    print(f"  Melody MIDI      → {mel_midi_path}")
    print(f"  Chord MIDI       → {chord_midi_path}")
    print(f"  Arrangement MIDI → {arr_midi_path}")

    # ── Step 7: Audio rendering ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7 │ Rendering audio …")
    print("=" * 60)
    renderer = MusicRenderer(soundfont=args.soundfont, sample_rate=args.sr)
    wav_path  = output_dir / "song.wav"
    renderer.render_from_parts(
        melody_notes=notes,
        chords=chords,
        arr_tracks=tracks,
        output_path=wav_path,
        tempo=tempo,
        normalize=True,
        fade_out_sec=args.fade_out,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"✓ Pipeline complete in {elapsed:.1f}s")
    print("=" * 60)
    print(f"  Song WAV    → {wav_path}")
    print(f"  MIDI files  → {output_dir}/")
    print(f"  Notes       : {len(notes)}")
    print(f"  Chords      : {len(chords)}")
    print(f"  Tempo       : {tempo:.1f} BPM")
    print(f"  Key         : {PITCH_CLASSES[features.key_root_pc]} {features.key_scale}")

    return {
        "notes":     notes,
        "chords":    chords,
        "tracks":    tracks,
        "midi_path": str(arr_midi_path),
        "wav_path":  str(wav_path),
    }


# ─── Training dispatch ────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace):
    """Dispatch to the appropriate training script."""
    if args.train == "transcription":
        from training.train_transcription import main as train_main
        sys.argv = [
            "train_transcription.py",
            f"--maestro_root={args.maestro_root}",
            f"--checkpoint_dir={args.checkpoint_dir or 'checkpoints/transcription'}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--lr={args.lr}",
        ]
        train_main()

    elif args.train == "harmony":
        from training.train_harmony_model import main as train_main
        sys.argv = [
            "train_harmony_model.py",
            f"--lakh_root={args.lakh_root}",
            f"--checkpoint_dir={args.checkpoint_dir or 'checkpoints/harmony'}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
        ]
        train_main()

    elif args.train == "arrangement":
        from training.train_arrangement_model import main as train_main
        sys.argv = [
            "train_arrangement_model.py",
            f"--lakh_root={args.lakh_root}",
            f"--checkpoint_dir={args.checkpoint_dir or 'checkpoints/arrangement'}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
        ]
        train_main()

    elif args.train == "all":
        print("Training all models sequentially …")
        for model_name in ("transcription", "harmony", "arrangement"):
            args.train = model_name
            run_training(args)
    else:
        print(f"Unknown training target: {args.train}")


# ─── Demo mode (no audio file required) ──────────────────────────────────────

def run_demo(args: argparse.Namespace):
    """
    Generate a demo output using a synthetic humming signal.
    No audio file required.
    """
    import tempfile

    print(BANNER)
    print("Running DEMO mode (synthetic humming: C–E–G–C melody) …\n")

    sr = SR_DEFAULT
    # Simulate humming: C4 (261Hz) → E4 (329Hz) → G4 (392Hz) → C5 (523Hz)
    freqs    = [261.63, 329.63, 392.00, 523.25]
    note_dur = 0.75
    t_total  = note_dur * len(freqs)
    y        = np.zeros(int(t_total * sr), dtype=np.float32)

    for i, hz in enumerate(freqs):
        s = int(i * note_dur * sr)
        e = int((i + 1) * note_dur * sr)
        t = np.linspace(0, note_dur, e - s)
        env = np.where(t < 0.01, t / 0.01, np.where(t > note_dur - 0.05,
                       (note_dur - t) / 0.05, 1.0))
        y[s:e] = (np.sin(2 * np.pi * hz * t) * env * 0.5).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    save_audio(y, tmp_path, sr=sr)
    args.audio = tmp_path
    result = run_pipeline(args)

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)
    return result


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hum-to-Music AI System",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "audio", nargs="?", default=None,
        help="Path to humming audio file (.wav/.mp3/.ogg). "
             "Omit to run demo with synthetic humming.",
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic humming audio (no file needed)")

    # ── Training mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--train",
        choices=["transcription", "harmony", "arrangement", "all"],
        default=None,
        help="Train a specific model instead of running inference",
    )
    parser.add_argument("--maestro_root",   default="data/raw/maestro")
    parser.add_argument("--lakh_root",      default="data/raw/lakh")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)

    # ── Model checkpoints ───────────────────────────────────────────────────
    parser.add_argument("--transcription_ckpt", default=None,
                        help="MelodyTranscriber checkpoint (.pt)")
    parser.add_argument("--harmony_ckpt",       default=None,
                        help="HarmonyGenerator checkpoint (.pt)")
    parser.add_argument("--arrangement_ckpt",   default=None,
                        help="ArrangementModel checkpoint (.pt)")

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save MIDI and WAV outputs")
    parser.add_argument("--soundfont",  default=None,
                        help="Path to .sf2 SoundFont for audio rendering")
    parser.add_argument("--sr",         type=int,   default=44100,
                        help="Output audio sample rate")
    parser.add_argument("--fade_out",   type=float, default=2.0,
                        help="Fade-out duration in seconds")

    # ── Pipeline options ─────────────────────────────────────────────────────
    parser.add_argument("--tempo",        type=float, default=None,
                        help="Override tempo in BPM (default: auto-detect)")
    parser.add_argument("--pitch_method", default="pyin",
                        choices=["pyin", "yin", "crepe"],
                        help="Pitch detection algorithm")
    parser.add_argument("--beam_size",    type=int,   default=1,
                        help="Beam search width for transcription")
    parser.add_argument("--temperature",  type=float, default=0.9,
                        help="Sampling temperature for neural models")
    parser.add_argument("--top_k",        type=int,   default=50,
                        help="Top-k filter for arrangement generation")
    parser.add_argument("--expand_song",  action="store_true",
                        help="Expand short melody into a full song structure")
    parser.add_argument("--song_style",   default="pop",
                        choices=["pop", "ballad", "minimal", "jazz"],
                        help="Song structure template")
    parser.add_argument("--verbose",      action="store_true",
                        help="Print detailed note / chord lists")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Training mode ─────────────────────────────────────────────────────
    if args.train:
        run_training(args)
        return

    # ── Demo mode ──────────────────────────────────────────────────────────
    if args.demo or args.audio is None:
        run_demo(args)
        return

    # ── Inference mode ─────────────────────────────────────────────────────
    if not Path(args.audio).exists():
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    run_pipeline(args)


if __name__ == "__main__":
    main()
