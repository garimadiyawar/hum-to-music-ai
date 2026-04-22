import { useCallback, useEffect, useRef, useState } from 'react';
import { Audio, AVPlaybackStatus } from 'expo-av';
import * as FileSystem from 'expo-file-system';

// ─── Types ────────────────────────────────────────────────────────────────────

export type RecorderState =
  | 'idle'        // nothing recorded yet
  | 'recording'   // actively recording
  | 'paused'      // recording paused
  | 'recorded'    // recording complete, ready to play / generate
  | 'playing';    // playing back the recording

export interface RecorderAPI {
  state:          RecorderState;
  durationMs:     number;    // current recording duration in ms
  playbackMs:     number;    // current playback position in ms
  isRecording:    boolean;
  hasRecording:   boolean;
  recordingUri:   string | null;

  startRecording:  () => Promise<void>;
  stopRecording:   () => Promise<void>;
  pauseRecording:  () => Promise<void>;
  resumeRecording: () => Promise<void>;
  playRecording:   () => Promise<void>;
  pausePlayback:   () => Promise<void>;
  deleteRecording: () => Promise<void>;
  permissionError: string | null;
}

// ─── Recording options ────────────────────────────────────────────────────────
// WAV 44.1 kHz mono as requested.

const RECORDING_OPTIONS: Audio.RecordingOptions = {
  android: {
    extension:   '.wav',
    outputFormat: Audio.AndroidOutputFormat.DEFAULT,
    audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
    sampleRate:   44100,
    numberOfChannels: 1,
    bitRate:      128000,
  },
  ios: {
    extension:         '.wav',
    outputFormat:      Audio.IOSOutputFormat.LINEARPCM,
    audioQuality:      Audio.IOSAudioQuality.HIGH,
    sampleRate:        44100,
    numberOfChannels:  1,
    bitRate:           128000,
    linearPCMBitDepth: 16,
    linearPCMIsBigEndian: false,
    linearPCMIsFloat:     false,
  },
  web: {
    mimeType:  'audio/wav',
    bitsPerSecond: 128000,
  },
};

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useRecorder(): RecorderAPI {
  const [state, setState]               = useState<RecorderState>('idle');
  const [durationMs, setDurationMs]     = useState(0);
  const [playbackMs, setPlaybackMs]     = useState(0);
  const [recordingUri, setRecordingUri] = useState<string | null>(null);
  const [permissionError, setPermErr]   = useState<string | null>(null);

  const recordingRef = useRef<Audio.Recording | null>(null);
  const soundRef     = useRef<Audio.Sound | null>(null);
  const timerRef     = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      _clearTimer();
      recordingRef.current?.stopAndUnloadAsync().catch(() => {});
      soundRef.current?.unloadAsync().catch(() => {});
    };
  }, []);

  // ── Timer helpers ─────────────────────────────────────────────────────────
  function _startTimer() {
    _clearTimer();
    timerRef.current = setInterval(() => {
      recordingRef.current?.getStatusAsync().then((s) => {
        if (s.isRecording || s.isDoneRecording) {
          setDurationMs(s.durationMillis ?? 0);
        }
      });
    }, 100);
  }

  function _clearTimer() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  // ── Permission ────────────────────────────────────────────────────────────
  async function _requestPermission(): Promise<boolean> {
    const { status } = await Audio.requestPermissionsAsync();
    if (status !== 'granted') {
      setPermErr('Microphone permission is required to record your hum.');
      return false;
    }
    setPermErr(null);
    return true;
  }

  // ── Audio session setup ───────────────────────────────────────────────────
  async function _configureAudio() {
    await Audio.setAudioModeAsync({
      allowsRecordingIOS:         true,
      playsInSilentModeIOS:       true,
      staysActiveInBackground:    false,
      shouldDuckAndroid:          true,
      playThroughEarpieceAndroid: false,
    });
  }

  // ── Recording ─────────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    const allowed = await _requestPermission();
    if (!allowed) return;

    // Unload any previous sound
    if (soundRef.current) {
      await soundRef.current.unloadAsync();
      soundRef.current = null;
    }
    setPlaybackMs(0);
    setDurationMs(0);

    await _configureAudio();

    const recording = new Audio.Recording();
    await recording.prepareToRecordAsync(RECORDING_OPTIONS);
    await recording.startAsync();

    recordingRef.current = recording;
    setState('recording');
    _startTimer();
  }, []);

  const stopRecording = useCallback(async () => {
    if (!recordingRef.current) return;
    _clearTimer();

    try {
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      recordingRef.current = null;

      if (uri) {
        // Copy to a stable filename in app's documents directory
        const dest = FileSystem.documentDirectory + 'humming.wav';
        await FileSystem.copyAsync({ from: uri, to: dest });
        setRecordingUri(dest);
      }
      setState('recorded');
    } catch {
      setState('idle');
    }
  }, []);

  const pauseRecording = useCallback(async () => {
    if (!recordingRef.current) return;
    await recordingRef.current.pauseAsync();
    _clearTimer();
    setState('paused');
  }, []);

  const resumeRecording = useCallback(async () => {
    if (!recordingRef.current) return;
    await recordingRef.current.startAsync();
    _startTimer();
    setState('recording');
  }, []);

  // ── Playback ──────────────────────────────────────────────────────────────

  const playRecording = useCallback(async () => {
    if (!recordingUri) return;

    // Switch audio session to playback mode
    await Audio.setAudioModeAsync({
      allowsRecordingIOS:   false,
      playsInSilentModeIOS: true,
    });

    if (soundRef.current) {
      const status = await soundRef.current.getStatusAsync();
      if (status.isLoaded) {
        await soundRef.current.replayAsync();
        setState('playing');
        return;
      }
    }

    const { sound } = await Audio.Sound.createAsync(
      { uri: recordingUri },
      { shouldPlay: true, volume: 1.0 },
      (status: AVPlaybackStatus) => {
        if (!status.isLoaded) return;
        setPlaybackMs(status.positionMillis ?? 0);
        if (status.didJustFinish) {
          setState('recorded');
          setPlaybackMs(0);
        }
      },
    );
    soundRef.current = sound;
    setState('playing');
  }, [recordingUri]);

  const pausePlayback = useCallback(async () => {
    if (!soundRef.current) return;
    await soundRef.current.pauseAsync();
    setState('recorded');
  }, []);

  // ── Delete ────────────────────────────────────────────────────────────────

  const deleteRecording = useCallback(async () => {
    _clearTimer();
    if (soundRef.current) {
      await soundRef.current.unloadAsync();
      soundRef.current = null;
    }
    if (recordingRef.current) {
      await recordingRef.current.stopAndUnloadAsync().catch(() => {});
      recordingRef.current = null;
    }
    if (recordingUri) {
      await FileSystem.deleteAsync(recordingUri, { idempotent: true });
    }
    setRecordingUri(null);
    setDurationMs(0);
    setPlaybackMs(0);
    setState('idle');
  }, [recordingUri]);

  return {
    state,
    durationMs,
    playbackMs,
    isRecording: state === 'recording',
    hasRecording: state === 'recorded' || state === 'playing',
    recordingUri,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    playRecording,
    pausePlayback,
    deleteRecording,
    permissionError,
  };
}

// ─── Formatter ────────────────────────────────────────────────────────────────

export function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min      = Math.floor(totalSec / 60);
  const sec      = totalSec % 60;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}
