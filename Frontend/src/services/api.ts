import axios, { AxiosError } from 'axios';

// ─── Configuration ────────────────────────────────────────────────────────────
// EXPO_PUBLIC_API_URL is set in your .env file at the project root.
// It must be your Mac's LAN IP address so a real iPhone can reach it.
//
//   .env:
//     EXPO_PUBLIC_API_URL=http://192.168.1.42:8000
//
// For the iOS Simulator (not a real device) you can use http://localhost:8000

const BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';
console.log("API BASE URL:", BASE_URL);

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000,   // 2 min — AI generation can take a while
});

// ─── Types ────────────────────────────────────────────────────────────────────

export interface GenerateResponse {
  song_url:  string;          // absolute URL to the generated WAV
  duration?: number;          // seconds
  key?:      string;          // e.g. "C major"
  tempo?:    number;          // BPM
}

export type UploadProgress = (percent: number) => void;

// ─── API calls ────────────────────────────────────────────────────────────────

/**
 * Upload a humming WAV and request a harmonised composition.
 *
 * The backend pipeline runs:
 *   WAV → pitch detection → melody notes → chord progression
 *       → multi-track arrangement → rendered WAV
 *
 * @param fileUri    Local file:// URI produced by expo-av
 * @param onProgress Upload progress callback (0–100)
 */
export async function generateMusic(
  fileUri: string,
  onProgress?: UploadProgress,
): Promise<GenerateResponse> {
  const formData = new FormData();

  // React Native's FormData accepts this object shape for binary files.
  // The backend receives it as an UploadFile with filename "humming.wav".
  formData.append('file', {
    uri:  fileUri,
    name: 'humming.m4a',
    type: 'audio/m4a',
  } as any);

  const response = await client.post<GenerateResponse>('/generate', formData, {
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    },
  });

  return response.data;
}

/**
 * Health-check — call on app start to verify the backend is reachable.
 * Returns true if the server responds, false otherwise.
 */
export async function pingServer(): Promise<boolean> {
  try {
    await client.get('/health', { timeout: 5_000 });
    return true;
  } catch {
    return false;
  }
}

// ─── Error helper ─────────────────────────────────────────────────────────────

export function getErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    if (!err.response) {
      return `Cannot reach server at ${BASE_URL}. Check your .env and Wi-Fi.`;
    }
    const status = err.response.status;
    if (status === 413) return 'Recording is too large. Try a shorter hum.';
    if (status === 422) return err.response.data?.detail ?? 'Could not process audio.';
    if (status >= 500)  return err.response.data?.detail ?? 'Server error — try again.';
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Something went wrong.';
}
