
# 🎵 Hum-to-Music AI

Turn a short humming recording into a **multi-instrument musical composition**.

This project builds a modular pipeline that converts a vocal melody into a full arrangement using **audio analysis, music theory, and neural network models**.

```

Humming Audio
↓
Pitch Detection
↓
Melody Transcription
↓
Chord Generation
↓
Multi-Instrument Arrangement
↓
MIDI Export + Audio Rendering

````

The system outputs both **MIDI files and a rendered WAV song**.

---

# Demo

Run the full pipeline on a humming recording:

```bash
python main.py humming.wav
````

Output:

```
outputs/
 ├── melody.mid
 ├── chords.mid
 ├── arrangement.mid
 └── song.wav
```

The system automatically:

* detects pitch from the humming
* converts it to musical notes
* generates a chord progression
* creates a multi-instrument arrangement
* renders the result to audio

---

# Example Pipeline Output

Example run:

```
Detected notes: 5
Generated chords: 3
Tempo: 112 BPM
Key: D minor
Tracks generated:
  melody
  bass
  piano
  strings
  pad
  drums
```

---

# Project Architecture

```
project/
│
├── configs/
│   └── model_config.yaml
│
├── data/
│   ├── humming_preprocessing.py
│   ├── dataset_loader.py
│   └── midi_processing.py
│
├── models/
│   ├── audio_encoder.py
│   ├── melody_transcriber.py
│   ├── harmony_generator.py
│   ├── arrangement_model.py
│   └── composition_transformer.py
│
├── inference/
│   ├── hum_to_melody.py
│   ├── melody_to_chords.py
│   ├── arrangement_generator.py
│   └── render_music.py
│
├── training/
│   ├── train_transcription.py
│   ├── train_harmony_model.py
│   └── train_arrangement_model.py
│
├── utils/
│   ├── audio_utils.py
│   ├── pitch_detection.py
│   ├── midi_utils.py
│   └── music_theory.py
│
└── main.py
```

---

# Core Components

## 1. Audio Processing

The system extracts musical information from humming using:

* mel-spectrogram analysis
* pitch detection (YIN / pYIN / CREPE)
* note segmentation
* key estimation
* tempo detection

Libraries used include **librosa** and **CREPE**.

---

## 2. Melody Transcription

Transforms the audio pitch signal into discrete note events.

Architecture:

```
Mel Spectrogram
      ↓
CNN Encoder
      ↓
Transformer
      ↓
MIDI Note Tokens
```

If no trained model checkpoint exists, the system falls back to a signal-processing transcription pipeline.

---

## 3. Harmony Generation

Predicts chords from the melody.

```
Melody Tokens
      ↓
Transformer Encoder
      ↓
Chord Predictions
```

Fallback rule-based harmony uses:

* detected key
* scale-compatible chords
* common progressions

---

## 4. Arrangement Generation

Creates a full arrangement from melody + chords.

Generated tracks:

* melody
* bass
* piano
* strings
* pad
* drums

The system can use either:

* a sequence model
* or a deterministic rule-based composer.

---

## 5. Audio Rendering

The MIDI arrangement can be rendered using:

1. SoundFonts (recommended)

or

2. a pure-Python synthesizer fallback.

For high-quality audio rendering install **FluidSynth**.

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Optional (better audio rendering):

Install **FluidSynth** and download a `.sf2` SoundFont.

---

# Running the Demo

Run with a humming recording:

```bash
python main.py humming.wav
```

Optional parameters:

```
--pitch_method crepe
--tempo 120
--song_style pop
--output_dir results
```

---

# Training Models

The repository includes training pipelines for three models:

| Model              | Task                     |
| ------------------ | ------------------------ |
| Melody Transcriber | audio → melody           |
| Harmony Generator  | melody → chords          |
| Arrangement Model  | melody + chords → tracks |

Datasets commonly used:

| Dataset   |
| --------- |
| MAESTRO   |
| Lakh MIDI |
| MedleyDB  |

Example training command:

```bash
python main.py --train transcription
```

Checkpoints are saved to:

```
checkpoints/
```

---

# Dependencies

Core libraries:

* PyTorch
* torchaudio
* librosa
* pretty_midi
* soundfile
* PyYAML
* tensorboard
* tqdm

Optional:

* CREPE (neural pitch detection)
* FluidSynth (audio rendering)

---

# License

MIT License.

---

# Motivation

Most music AI tools require **MIDI input or symbolic music**.

This project explores the harder problem:

**converting raw human humming into structured music.**

It combines **audio signal processing, music theory, and machine learning** into a unified pipeline.

