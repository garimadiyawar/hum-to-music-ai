# 🎧 Hum-to-Music AI

Turn a simple hummed melody into a fully arranged musical composition using AI.

The system converts human humming into structured music through pitch detection, melody extraction, harmony generation, arrangement, and audio rendering — all served through a FastAPI backend and a React Native (Expo) mobile app.

---

## ✨ Features

- 🎤 Record humming from mobile
- 🧠 AI-based melody extraction
- 🎼 Automatic chord generation
- 🎹 Multi-instrument arrangement pipeline
- 🔊 Audio rendering (WAV output)
- 📱 Real-time mobile preview via Expo Go
- 🌐 REST API backend (FastAPI)

---

## 🧠 Architecture

```

Frontend (React Native + Expo)
↓
POST /generate (audio upload)
↓
FastAPI Backend
├── Hum → Melody
├── Melody → Chords
├── Arrangement Generator
├── Music Renderer
↓
WAV file + metadata
↓
Static file served (/output)
↓
Frontend playback

````

---

## 🛠 Tech Stack

### Backend
- FastAPI
- PyTorch
- Librosa
- PrettyMIDI
- NumPy
- Pydub

### Frontend
- React Native (Expo)
- Axios
- Expo AV / Audio APIs

---

## 🚀 Setup Instructions

### 1. Backend

```bash
cd Backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python server.py
````

Run server:

```
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

### 2. Frontend

```bash
cd Frontend
npm install
npx expo start --clear
```

Make sure `.env` contains:

```
EXPO_PUBLIC_API_URL=http://<YOUR_LAPTOP_IP>:8000
```

---

## 📁 Project Structure

```
hum-to-music-ai/
│
├── Backend/
│   ├── server.py
│   ├── inference/
│   ├── utils/
│   ├── server_outputs/
│   ├── server_uploads/
│   └── .env
│
├── Frontend/
│   ├── App.js / screens/
│   ├── api.ts
│   ├── .env
│   └── package.json
│
└── README.md
```

---

## ⚠️ Notes

* Must run both devices on the same WiFi network
* Use LAN IP (not localhost) for API connection
* Windows Firewall may block ports 8000 and 8081
* Audio rendering is CPU-based (no GPU required for inference stage)

---

## 🧪 API Endpoints

### Health Check

```
GET /health
```

### Generate Music

```
POST /generate
FormData: file (audio/wav or audio/m4a)
```

Response:

```json
{
  "song_url": "http://<ip>:8000/output/file.wav",
  "duration": 12.4,
  "key": "C major",
  "tempo": 110
}
```

---

## 💡 Future Improvements

* Replace sine renderer with real instrument soundfonts
* Add real-time waveform visualization
* Improve pitch detection robustness
* Add genre conditioning (lofi, cinematic, pop)
* Cloud deployment (Render / AWS)

---

## ⚡ Status

MVP working:
✔ Audio recording
✔ Backend pipeline
✔ Music generation
✔ File serving

In progress:
⚠ Network stability (Expo + LAN setup)

````

---

# 🧼 3. Small repo cleanup advice (important but quick)

Before pushing:

### Do this:

```bash
git status
````

Make sure you are NOT about to commit:

* `venv/`
* `node_modules/`
* `server_outputs/`
* `.env`

If you see them staged → remove:

```bash
git rm -r --cached venv
git rm -r --cached node_modules
```

---

# 🚀 When you're ready to push

```bash
git add .
git commit -m "Initial working MVP: hum-to-music AI pipeline + mobile integration"
git push origin main
```