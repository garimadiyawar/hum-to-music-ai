"""
server.py
─────────
FastAPI HTTP server that wraps the hum-to-music AI pipeline and
exposes the two endpoints the React Native app calls:

    POST /generate   multipart WAV upload → JSON with song_url
    GET  /health     liveness probe

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

The --host 0.0.0.0 is essential — it makes the server reachable from
devices on the same Wi-Fi network (i.e. your iPhone running Expo Go).
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydub import AudioSegment

# ── Make sure our project modules are importable ──────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from inference.hum_to_melody       import HumToMelody
from inference.melody_to_chords    import MelodyToChords
from inference.arrangement_generator import ArrangementGenerator
from inference.render_music        import MusicRenderer
from utils.midi_utils              import arrangement_to_midi, save_midi
from utils.music_theory            import PITCH_CLASSES


# ─── Directories ─────────────────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "server_uploads"
OUTPUT_DIR = ROOT / "server_outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Load pipeline (once at startup, not per request) ────────────────────────
# Checkpoints are optional — the pipeline falls back to signal-processing
# and rule-based generation if they don't exist.

print("Loading pipeline modules…")

hum_to_melody = HumToMelody(
    checkpoint=os.environ.get("TRANSCRIPTION_CKPT"),   # e.g. checkpoints/transcription/best.pt
    pitch_method=os.environ.get("PITCH_METHOD", "pyin"),
)

melody_to_chords = MelodyToChords(
    checkpoint=os.environ.get("HARMONY_CKPT"),         # e.g. checkpoints/harmony/best.pt
)

arrangement_gen = ArrangementGenerator(
    checkpoint=os.environ.get("ARRANGEMENT_CKPT"),     # e.g. checkpoints/arrangement/best.pt
)

renderer = MusicRenderer(
    soundfont=os.environ.get("SF2_PATH"),              # e.g. assets/GeneralUser_GS.sf2
    sample_rate=int(os.environ.get("SAMPLE_RATE", "44100")),
)

print("Pipeline ready.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Hum-to-Music API", version="1.0.0")

# CORS — allow requests from Expo Go / local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated WAV files as static assets so the app can stream them
app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output",
)


# ─── Response model ───────────────────────────────────────────────────────────

class GenerateResponse(BaseModel):
    song_url:  str
    duration:  float | None = None
    key:       str | None   = None
    tempo:     float | None = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — the app pings this on startup."""
    return {"status": "ok"}

@app.post("/debug")
async def debug(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }

@app.post("/generate", response_model=GenerateResponse)
async def generate(file: UploadFile = File(...)):
    """
    Accept a humming WAV file and return a harmonised composition.

    Expects:  multipart/form-data  field name: "file"
    Returns:  { song_url, duration, key, tempo }
    """
    # ── Validate upload ───────────────────────────────────────────────────────
    if not file.content_type.startswith("audio"):
        raise HTTPException(status_code=422, detail="Uploaded file must be audio.")
    
    content = await file.read()
    if len(content) < 1_000:
        raise HTTPException(status_code=422, detail="Audio file is too short.")
    if len(content) > 50 * 1_024 * 1_024:   # 50 MB cap
        raise HTTPException(status_code=413, detail="Audio file is too large.")

    # ── Save upload ───────────────────────────────────────────────────────────
    job_id    = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".audio"
    input_path = UPLOAD_DIR / f"{job_id}_humming{ext}"
    input_path.write_bytes(content)
    # Convert phone audio to WAV for the pipeline
    if input_path.suffix.lower() != ".wav":
        wav_path = UPLOAD_DIR / f"{job_id}_humming.wav"

        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        audio.export(wav_path, format="wav")
        input_path = wav_path

        
    try:
        t0 = time.perf_counter()

        # ── Stage 1: melody transcription ─────────────────────────────────────
        notes, features = hum_to_melody.transcribe_with_features(str(input_path))
        if not notes:
            raise HTTPException(status_code=422, detail="Could not detect a melody. Try humming more clearly.")

        tempo       = float(features.tempo_bpm)
        key_root_pc = int(features.key_root_pc)
        key_scale   = features.key_scale
        duration    = float(max(n["end"] for n in notes))

        # ── Stage 2: harmony ──────────────────────────────────────────────────
        chords = melody_to_chords.harmonize(notes, duration=duration, tempo=tempo)

        # ── Stage 3: arrangement ──────────────────────────────────────────────
        tracks = arrangement_gen.arrange(
            melody_notes=notes,
            chords=chords,
            tempo=tempo,
            duration=duration,
            key_root_pc=key_root_pc,
            key_scale=key_scale,
        )

        # ── Stage 4: render ───────────────────────────────────────────────────
        wav_path = OUTPUT_DIR / f"{job_id}_song.wav"
        renderer.render_from_parts(
            melody_notes=notes,
            chords=chords,
            arr_tracks=tracks,
            output_path=str(wav_path),
            tempo=tempo,
            normalize=True,
            fade_out_sec=2.0,
        )
        # ── Build public URL ──────────────────────────────────────────────────
        # SERVER_HOST is the LAN IP of this machine, so the iPhone can reach it.
        # Falls back to localhost (works for iOS Simulator only).
        host     = os.environ.get("SERVER_HOST", "localhost")
        port     = int(os.environ.get("PORT", "8000"))
        song_url = f"http://{host}:{port}/output/{wav_path.name}"

        key_name = f"{PITCH_CLASSES[key_root_pc]} {key_scale.replace('natural_', '')}"
        
        # Also save MIDI alongside (optional, nice for debugging)
        midi_path = OUTPUT_DIR / f"{job_id}_arrangement.mid"
        pm        = arrangement_to_midi(tracks, tempo=tempo)
        save_midi(pm, str(midi_path))

        elapsed = time.perf_counter() - t0
        print(f"[{job_id}] Done in {elapsed:.1f}s — "
              f"{len(notes)} notes, {len(chords)} chords, tempo={tempo:.0f}")

        return GenerateResponse(
            song_url=song_url,
            duration=duration,
            key=key_name,
            tempo=tempo,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"[{job_id}] Pipeline error: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(exc)}")
    finally:
        # Clean up the raw upload (keep output for playback)
        input_path.unlink(missing_ok=True)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=True)
