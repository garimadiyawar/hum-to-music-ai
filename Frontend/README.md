# 🎵 Hum — React Native App

Minimal humming-to-music interface. Record a melody hum, send it to your AI
pipeline, receive and play the harmonized composition.

## Design

Retro analog warmth: chocolate browns, warm creams, monospaced type — like
recording onto cassette tape but inside a minimal iOS app.

## Quick start

```bash
cd humming-app
npm install
npx expo start
```

Scan the QR code with **Expo Go** on your iPhone.

## Environment

Create a `.env` file (or set the variable in `eas.json`):

```
EXPO_PUBLIC_API_URL=http://192.168.1.XX:8000
```

Replace `192.168.1.XX` with your Mac's LAN IP address so the iPhone on
the same Wi-Fi network can reach your Python server.

Find your Mac's IP: **System Settings → Wi-Fi → Details → IP Address**

Your Python backend should expose:

```
POST /generate          multipart/form-data  file=humming.wav
→ { "song_url": "http://...", "duration": 14.2, "key": "C major", "tempo": 120 }

GET  /health            → 200 OK
```

## File structure

```
App.tsx                        Root navigator (React Navigation stack)
src/
  styles/theme.ts              Colours, typography, spacing — single source of truth
  hooks/
    useRecorder.ts             All recording/playback state (expo-av)
    useToast.ts                Toast notification system
  services/api.ts              Axios client → backend
  components/
    RecorderControls.tsx       Pulsing record button + duration display
    PlaybackControls.tsx       Waveform bar + play/pause/delete
    GenerateButton.tsx         Retro pixel-shadow CTA button
    ToastMessage.tsx           Animated status messages
  screens/
    HomeScreen.tsx             Record → play → generate flow
    ResultScreen.tsx           Animated waveform player for generated song
```

## States

| State       | Description                              |
|-------------|------------------------------------------|
| `idle`      | Nothing recorded                         |
| `recording` | Mic is active, button pulses             |
| `paused`    | Recording paused                         |
| `recorded`  | Recording saved, ready to play/generate  |
| `playing`   | Playback of the hum recording            |

## Dependencies

| Package                        | Purpose                         |
|--------------------------------|---------------------------------|
| expo-av                        | Recording + playback            |
| expo-file-system               | Save WAV to local storage       |
| axios                          | HTTP requests                   |
| @react-navigation/native-stack | Screen navigation               |
| react-native-reanimated        | Smooth animations               |
| react-native-gesture-handler   | Required by navigation          |

## Backend contract

```http
POST /generate HTTP/1.1
Content-Type: multipart/form-data

file: <binary WAV data>  (filename: humming.wav)
```

Response:
```json
{
  "song_url": "https://yourserver.com/output/song_abc123.wav",
  "duration": 24.5,
  "key":      "G major",
  "tempo":    118
}
```

All fields except `song_url` are optional.
