import axios, { AxiosError } from 'axios';

// ─── Configuration ────────────────────────────────────────────────────────────
// Replace BASE_URL with your server address when testing on a real device.
// For local development with Expo Go on iPhone, use your Mac's LAN IP:
//   e.g. http://192.168.1.42:8000
// For Android emulator use http://10.0.2.2:8000
// For iOS simulator use http://localhost:8000

const BASE_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000';

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120_000, // 2 min — generation can take a while
});

// ─── Types ────────────────────────────────────────────────────────────────────

export interface GenerateResponse {
  song_url: string;       // URL to the generated WAV
  duration?: number;      // seconds (optional metadata)
  key?: string;           // e.g. "C major"
  tempo?: number;         // BPM
}

export type UploadProgress = (percent: number) => void;

// ─── API calls ────────────────────────────────────────────────────────────────

/**
 * Upload a humming recording and request a harmonized composition.
 *
 * @param fileUri   Local file:// URI from expo-av / expo-file-system
 * @param onProgress  Optional upload progress callback (0–100)
 */
export async function generateMusic(
  fileUri: string,
  onProgress?: UploadProgress,
): Promise<GenerateResponse> {
  const formData = new FormData();

  // React Native FormData accepts the object form for files
  formData.append('file', {
    uri:  fileUri,
    name: 'humming.wav',
    type: 'audio/wav',
  } as unknown as Blob);

  const response = await client.post<GenerateResponse>('/generate', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onProgress && evt.total) {
        onProgress(Math.round((evt.loaded / evt.total) * 100));
      }
    },
  });

  return response.data;
}

/**
 * Health-check ping — useful to verify connectivity before recording.
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
    if (!err.response) return 'Cannot reach the server. Check your network.';
    const status = err.response.status;
    if (status === 413) return 'Recording is too large. Try a shorter hum.';
    if (status === 422) return 'Could not process audio. Try again.';
    if (status >= 500) return 'Server error. The AI is taking a nap — try again.';
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Something went wrong.';
}
