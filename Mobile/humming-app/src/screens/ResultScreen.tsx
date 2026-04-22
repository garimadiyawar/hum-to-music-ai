import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  SafeAreaView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Audio, AVPlaybackStatus } from 'expo-av';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';

import ToastMessage from '../components/ToastMessage';
import { useToast }  from '../hooks/useToast';
import { Colors, Spacing, Typography, Shadows } from '../styles/theme';
import { RootStackParamList } from '../../App';
import { formatDuration } from '../hooks/useRecorder';

// ─── Types ────────────────────────────────────────────────────────────────────

type NavProp   = NativeStackNavigationProp<RootStackParamList, 'Result'>;
type RoutePropT = RouteProp<RootStackParamList, 'Result'>;

interface Props {
  navigation: NavProp;
  route:      RoutePropT;
}

type PlayState = 'idle' | 'loading' | 'playing' | 'paused' | 'finished';

// ─── Screen ───────────────────────────────────────────────────────────────────

export default function ResultScreen({ navigation, route }: Props) {
  const { songUrl, duration, key: songKey, tempo } = route.params;
  const toast = useToast();

  const [playState,   setPlayState]   = useState<PlayState>('idle');
  const [positionMs,  setPositionMs]  = useState(0);
  const [durationMs,  setDurationMs]  = useState((duration ?? 0) * 1000);
  const soundRef                       = useRef<Audio.Sound | null>(null);

  // ── Entry animations ──────────────────────────────────────────────────────
  const cardScale   = useRef(new Animated.Value(0.88)).current;
  const cardOpacity = useRef(new Animated.Value(0)).current;
  const cardY       = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.spring(cardScale,   { toValue: 1,   useNativeDriver: true, bounciness: 6 }),
      Animated.timing(cardOpacity, { toValue: 1, duration: 400, useNativeDriver: true }),
      Animated.timing(cardY,       { toValue: 0, duration: 450, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  }, [cardOpacity, cardScale, cardY]);

  // ── Cleanup ───────────────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      soundRef.current?.unloadAsync().catch(() => {});
    };
  }, []);

  // ── Playback status callback ──────────────────────────────────────────────
  const onPlaybackStatus = useCallback((status: AVPlaybackStatus) => {
    if (!status.isLoaded) return;
    setPositionMs(status.positionMillis ?? 0);
    if (status.durationMillis) setDurationMs(status.durationMillis);
    if (status.didJustFinish) {
      setPlayState('finished');
      setPositionMs(0);
    }
  }, []);

  // ── Load & play ───────────────────────────────────────────────────────────
  const loadAndPlay = useCallback(async () => {
    setPlayState('loading');
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS:   false,
        playsInSilentModeIOS: true,
      });

      if (soundRef.current) {
        await soundRef.current.unloadAsync();
        soundRef.current = null;
      }

      const { sound } = await Audio.Sound.createAsync(
        { uri: songUrl },
        { shouldPlay: true, volume: 1.0 },
        onPlaybackStatus,
      );
      soundRef.current = sound;
      setPlayState('playing');
    } catch (err) {
      setPlayState('idle');
      toast.show('Could not load audio. Check your connection.', 'error');
    }
  }, [songUrl, onPlaybackStatus, toast]);

  // ── Toggle play/pause ─────────────────────────────────────────────────────
  const handlePlayPause = useCallback(async () => {
    if (playState === 'idle' || playState === 'finished') {
      await loadAndPlay();
      return;
    }
    if (!soundRef.current) return;

    if (playState === 'playing') {
      await soundRef.current.pauseAsync();
      setPlayState('paused');
    } else if (playState === 'paused') {
      await soundRef.current.playAsync();
      setPlayState('playing');
    }
  }, [playState, loadAndPlay]);

  // ── Replay from beginning ─────────────────────────────────────────────────
  const handleReplay = useCallback(async () => {
    if (!soundRef.current) {
      await loadAndPlay();
      return;
    }
    await soundRef.current.stopAsync();
    await soundRef.current.setPositionAsync(0);
    await soundRef.current.playAsync();
    setPlayState('playing');
  }, [loadAndPlay]);

  // ── Derived ───────────────────────────────────────────────────────────────
  const progress = durationMs > 0 ? positionMs / durationMs : 0;
  const isPlaying = playState === 'playing';
  const isLoading = playState === 'loading';

  const playIcon =
    isLoading  ? '…' :
    isPlaying  ? '‖' :
    '▶';

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>

        {/* ── Back button ───────────────────────────────────────────── */}
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backBtn}
          hitSlop={16}
        >
          <Text style={styles.backText}>← BACK</Text>
        </TouchableOpacity>

        {/* ── Card ──────────────────────────────────────────────────── */}
        <Animated.View
          style={[
            styles.card,
            {
              opacity:   cardOpacity,
              transform: [{ scale: cardScale }, { translateY: cardY }],
            },
          ]}
        >
          {/* Header */}
          <View style={styles.cardHeader}>
            <Text style={styles.cardTitle}>YOUR COMPOSITION</Text>
            <View style={styles.headerRule} />
          </View>

          {/* Metadata row */}
          <View style={styles.metaRow}>
            {songKey && (
              <MetaChip label="KEY" value={songKey} />
            )}
            {tempo && (
              <MetaChip label="BPM" value={String(Math.round(tempo))} />
            )}
            <MetaChip label="FORMAT" value="WAV" />
          </View>

          {/* Waveform bars */}
          <AnimatedWaveform isPlaying={isPlaying} progress={progress} />

          {/* Time */}
          <View style={styles.timeRow}>
            <Text style={styles.timeText}>{formatDuration(positionMs)}</Text>
            <Text style={styles.timeText}>{formatDuration(durationMs)}</Text>
          </View>

          {/* Controls */}
          <View style={styles.controls}>
            {/* Replay */}
            <TouchableOpacity onPress={handleReplay} style={styles.sideBtn} hitSlop={12}>
              <Text style={styles.sideBtnText}>↺</Text>
              <Text style={styles.sideBtnLabel}>REPLAY</Text>
            </TouchableOpacity>

            {/* Play / Pause */}
            <TouchableOpacity
              onPress={handlePlayPause}
              style={[styles.playBtn, isPlaying && styles.playBtnActive]}
              disabled={isLoading}
            >
              <Text style={[styles.playBtnIcon, isPlaying && styles.playBtnIconActive]}>
                {playIcon}
              </Text>
            </TouchableOpacity>

            {/* Placeholder for symmetry */}
            <View style={styles.sideBtn} />
          </View>
        </Animated.View>

        {/* ── Record again ──────────────────────────────────────────── */}
        <TouchableOpacity
          onPress={() => {
            soundRef.current?.unloadAsync().catch(() => {});
            navigation.goBack();
          }}
          style={styles.recordAgainBtn}
        >
          <Text style={styles.recordAgainText}>♪ HUM AGAIN</Text>
        </TouchableOpacity>

        {/* ── Decorative footer text ────────────────────────────────── */}
        <Text style={styles.footerText}>
          {'generated by ai ·  powered by your melody'}
        </Text>
      </View>

      <ToastMessage toast={toast.toast} opacity={toast.opacity} />
    </SafeAreaView>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function MetaChip({ label, value }: { label: string; value: string }) {
  return (
    <View style={chipStyles.chip}>
      <Text style={chipStyles.label}>{label}</Text>
      <Text style={chipStyles.value}>{value}</Text>
    </View>
  );
}

// Animated waveform (bars that sway while playing)
function AnimatedWaveform({ isPlaying, progress }: { isPlaying: boolean; progress: number }) {
  const BARS     = 40;
  const filledTo = Math.round(progress * BARS);

  // Per-bar animation values
  const anims = useRef(
    Array.from({ length: BARS }, () => new Animated.Value(0.4))
  ).current;

  useEffect(() => {
    if (!isPlaying) {
      anims.forEach((a) =>
        Animated.timing(a, { toValue: 0.4, duration: 300, useNativeDriver: false }).start()
      );
      return;
    }

    // Stagger a looping "breathe" per bar
    const loops = anims.map((a, i) =>
      Animated.loop(
        Animated.sequence([
          Animated.timing(a, {
            toValue:  0.5 + Math.random() * 0.5,
            duration: 300 + (i % 7) * 60,
            useNativeDriver: false,
          }),
          Animated.timing(a, {
            toValue:  0.2 + Math.random() * 0.3,
            duration: 300 + (i % 5) * 80,
            useNativeDriver: false,
          }),
        ]),
      )
    );
    loops.forEach((l, i) => setTimeout(() => l.start(), i * 18));
    return () => loops.forEach((l) => l.stop());
  }, [isPlaying, anims]);

  return (
    <View style={wfStyles.container}>
      {anims.map((anim, i) => {
        const baseH   = 6 + ((i * 13 + 7) % 17) * 1.6;   // deterministic pseudo-random height
        const filled  = i < filledTo;
        return (
          <Animated.View
            key={i}
            style={[
              wfStyles.bar,
              {
                height:          anim.interpolate({ inputRange: [0, 1], outputRange: [baseH * 0.4, baseH] }),
                backgroundColor: filled ? Colors.accent : Colors.highlight,
              },
            ]}
          />
        );
      })}
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe: {
    flex:            1,
    backgroundColor: Colors.background,
  },

  container: {
    flex:            1,
    paddingHorizontal: Spacing.xl,
    paddingTop:      Spacing.lg,
    paddingBottom:   Spacing.xl,
    alignItems:      'center',
    gap:              Spacing.lg,
  },

  backBtn: {
    alignSelf:  'flex-start',
  },

  backText: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXS,
    color:         Colors.inkLight,
    letterSpacing: Typography.letterSpacingWide,
  },

  // ── Card ──────────────────────────────────────────────────────────────────

  card: {
    width:          '100%',
    backgroundColor: Colors.surface,
    borderRadius:   6,
    padding:         Spacing.lg,
    gap:             Spacing.md,
    borderWidth:    1,
    borderColor:    Colors.borderLight,
    ...Shadows.deep,
  },

  cardHeader: {
    gap: Spacing.sm,
  },

  cardTitle: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXS,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
  },

  headerRule: {
    height:          2,
    backgroundColor: Colors.accent,
    width:           40,
  },

  metaRow: {
    flexDirection: 'row',
    gap:            Spacing.sm,
    flexWrap:      'wrap',
  },

  timeRow: {
    flexDirection:  'row',
    justifyContent: 'space-between',
  },

  timeText: {
    fontFamily: Typography.fontMono,
    fontSize:   Typography.sizeXS,
    color:      Colors.inkFaint,
  },

  controls: {
    flexDirection:  'row',
    alignItems:     'center',
    justifyContent: 'space-between',
    marginTop:       Spacing.sm,
  },

  sideBtn: {
    width:       52,
    alignItems: 'center',
    gap:          4,
  },

  sideBtnText: {
    fontSize: 22,
    color:    Colors.inkMid,
  },

  sideBtnLabel: {
    fontFamily:    Typography.fontMono,
    fontSize:      9,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
  },

  playBtn: {
    width:           72,
    height:          72,
    borderRadius:    36,
    borderWidth:     2,
    borderColor:     Colors.accent,
    alignItems:      'center',
    justifyContent:  'center',
    backgroundColor: Colors.background,
    ...Shadows.soft,
  },

  playBtnActive: {
    backgroundColor: Colors.accent,
  },

  playBtnIcon: {
    fontSize:   26,
    color:      Colors.accent,
    lineHeight: 30,
  },

  playBtnIconActive: {
    color: Colors.background,
  },

  // ── Footer buttons ────────────────────────────────────────────────────────

  recordAgainBtn: {
    paddingVertical:   Spacing.md,
    paddingHorizontal: Spacing.xl,
    borderWidth:       1,
    borderColor:       Colors.border,
    borderRadius:      4,
    width:             '100%',
    alignItems:        'center',
    marginTop:          Spacing.sm,
  },

  recordAgainText: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeMD,
    color:         Colors.inkMid,
    letterSpacing: Typography.letterSpacingWide,
  },

  footerText: {
    fontFamily: Typography.fontMono,
    fontSize:   9,
    color:      Colors.inkFaint,
    textAlign:  'center',
    letterSpacing: 1,
    marginTop:  'auto',
  },
});

const chipStyles = StyleSheet.create({
  chip: {
    paddingVertical:   4,
    paddingHorizontal: Spacing.sm,
    backgroundColor:  Colors.surfaceRaised,
    borderRadius:     2,
    borderWidth:      1,
    borderColor:      Colors.border,
    alignItems:       'center',
  },
  label: {
    fontFamily:    Typography.fontMono,
    fontSize:      8,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
  },
  value: {
    fontFamily: Typography.fontMono,
    fontSize:   Typography.sizeSM,
    color:      Colors.inkMid,
    fontWeight: Typography.weightBold,
  },
});

const wfStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems:    'flex-end',
    height:         50,
    gap:             2,
    width:          '100%',
  },
  bar: {
    flex:         1,
    borderRadius: 2,
  },
});
