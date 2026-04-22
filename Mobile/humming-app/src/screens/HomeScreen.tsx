import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Animated,
  Easing,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';

import RecorderControls from '../components/RecorderControls';
import PlaybackControls from '../components/PlaybackControls';
import GenerateButton   from '../components/GenerateButton';
import ToastMessage     from '../components/ToastMessage';

import { useRecorder } from '../hooks/useRecorder';
import { useToast }    from '../hooks/useToast';
import { generateMusic, getErrorMessage } from '../services/api';
import { Colors, Spacing, Typography } from '../styles/theme';
import { RootStackParamList } from '../../App';

// ─── Types ────────────────────────────────────────────────────────────────────

type NavProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

interface Props {
  navigation: NavProp;
}

// ─── Screen ───────────────────────────────────────────────────────────────────

export default function HomeScreen({ navigation }: Props) {
  const recorder = useRecorder();
  const toast    = useToast();

  const [generating, setGenerating]   = useState(false);
  const [uploadPct,   setUploadPct]   = useState(0);

  // ── Entry animation ───────────────────────────────────────────────────────
  const headerOpacity = useRef(new Animated.Value(0)).current;
  const headerY       = useRef(new Animated.Value(-20)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(headerOpacity, { toValue: 1, duration: 600, delay: 100, useNativeDriver: true }),
      Animated.timing(headerY,       { toValue: 0, duration: 600, delay: 100, easing: Easing.out(Easing.cubic), useNativeDriver: true }),
    ]).start();
  }, [headerOpacity, headerY]);

  // ── Permission error feedback ─────────────────────────────────────────────
  useEffect(() => {
    if (recorder.permissionError) {
      toast.show(recorder.permissionError, 'error');
    }
  }, [recorder.permissionError]);

  // ── Generate music ────────────────────────────────────────────────────────
  const handleGenerate = useCallback(async () => {
    if (!recorder.recordingUri) return;

    setGenerating(true);
    setUploadPct(0);

    try {
      const result = await generateMusic(recorder.recordingUri, (pct) => {
        setUploadPct(pct);
      });

      navigation.navigate('Result', {
        songUrl:   result.song_url,
        duration:  result.duration,
        key:       result.key,
        tempo:     result.tempo,
      });
    } catch (err) {
      toast.show(getErrorMessage(err), 'error');
    } finally {
      setGenerating(false);
      setUploadPct(0);
    }
  }, [recorder.recordingUri, navigation, toast]);

  // ── Derived booleans ──────────────────────────────────────────────────────
  const showPlayback = recorder.hasRecording || recorder.state === 'playing';
  const showGenerate = showPlayback && !generating;
  const canGenerate  = showPlayback && !generating && recorder.state !== 'playing';

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        {/* ── Header ─────────────────────────────────────────────────── */}
        <Animated.View
          style={[styles.header, { opacity: headerOpacity, transform: [{ translateY: headerY }] }]}
        >
          <Text style={styles.wordmark}>HUM</Text>
          <Text style={styles.tagline}>melody into music</Text>
          {/* Decorative pixel row */}
          <View style={styles.pixelRow}>
            {Array.from({ length: 8 }, (_, i) => (
              <View
                key={i}
                style={[styles.pixel, { opacity: 0.2 + (i % 3) * 0.25 }]}
              />
            ))}
          </View>
        </Animated.View>

        {/* ── Record controls ─────────────────────────────────────────── */}
        <View style={styles.recordSection}>
          <RecorderControls
            state={recorder.state}
            durationMs={recorder.durationMs}
            onStart={recorder.startRecording}
            onStop={recorder.stopRecording}
            onPause={recorder.pauseRecording}
            onResume={recorder.resumeRecording}
          />
        </View>

        {/* ── Playback controls (shown once something is recorded) ─────── */}
        {showPlayback && (
          <Animated.View style={styles.playbackSection}>
            <View style={styles.sectionDivider} />
            <Text style={styles.sectionLabel}>YOUR HUM</Text>
            <PlaybackControls
              state={recorder.state}
              durationMs={recorder.durationMs}
              playbackMs={recorder.playbackMs}
              onPlay={recorder.playRecording}
              onPause={recorder.pausePlayback}
              onDelete={recorder.deleteRecording}
            />
          </Animated.View>
        )}

        {/* ── Generate button ──────────────────────────────────────────── */}
        {showGenerate && (
          <View style={styles.generateSection}>
            <GenerateButton
              onPress={handleGenerate}
              loading={generating}
              disabled={!canGenerate}
              progress={uploadPct}
            />
          </View>
        )}

        {/* ── Idle hint ────────────────────────────────────────────────── */}
        {recorder.state === 'idle' && (
          <Text style={styles.hint}>
            {'tap the circle\nthen hum your melody'}
          </Text>
        )}
      </ScrollView>

      {/* ── Toast ─────────────────────────────────────────────────────── */}
      <ToastMessage toast={toast.toast} opacity={toast.opacity} />
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  safe: {
    flex:            1,
    backgroundColor: Colors.background,
  },

  scroll: {
    flexGrow:        1,
    paddingHorizontal: Spacing.xl,
    paddingTop:      Spacing.xl,
    paddingBottom:   Spacing.xxl,
    alignItems:      'center',
    gap:              Spacing.xl,
  },

  // ── Header
  header: {
    alignItems: 'center',
    gap:         Spacing.xs,
  },

  wordmark: {
    fontFamily:    Typography.fontDisplay,
    fontSize:      Typography.sizeHero,
    color:         Colors.inkMid,
    letterSpacing: 18,
    fontStyle:     'italic',
  },

  tagline: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXS,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
    textTransform: 'lowercase',
  },

  pixelRow: {
    flexDirection: 'row',
    gap:            4,
    marginTop:      Spacing.xs,
  },

  pixel: {
    width:           6,
    height:          6,
    backgroundColor: Colors.accent,
  },

  // ── Sections
  recordSection: {
    alignItems:  'center',
    paddingTop:   Spacing.lg,
  },

  playbackSection: {
    width:      '100%',
    alignItems: 'center',
    gap:         Spacing.md,
  },

  generateSection: {
    width:      '100%',
    alignItems: 'center',
    paddingTop: Spacing.sm,
  },

  sectionDivider: {
    width:           '100%',
    height:           1,
    backgroundColor: Colors.borderLight,
  },

  sectionLabel: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXS,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
    alignSelf:     'flex-start',
  },

  hint: {
    fontFamily:  Typography.fontDisplay,
    fontSize:    Typography.sizeSM,
    color:       Colors.inkFaint,
    textAlign:   'center',
    lineHeight:  22,
    fontStyle:   'italic',
    marginTop:   Spacing.xl,
  },
});
