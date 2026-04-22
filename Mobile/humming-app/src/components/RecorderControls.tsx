import React, { useEffect, useRef } from 'react';
import {
  Animated,
  Easing,
  Pressable,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Colors, RecordButton, Spacing, Typography } from '../styles/theme';
import { RecorderState, formatDuration } from '../hooks/useRecorder';

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  state:           RecorderState;
  durationMs:      number;
  onStart:         () => void;
  onStop:          () => void;
  onPause:         () => void;
  onResume:        () => void;
}

// ─── Pulse animation constants ────────────────────────────────────────────────

const PULSE_DURATION = 900;

// ─── Component ────────────────────────────────────────────────────────────────

export default function RecorderControls({
  state,
  durationMs,
  onStart,
  onStop,
  onPause,
  onResume,
}: Props) {
  const isRecording = state === 'recording';
  const isPaused    = state === 'paused';
  const isActive    = isRecording || isPaused;

  // ── Pulse animation ───────────────────────────────────────────────────────
  const pulseScale   = useRef(new Animated.Value(1)).current;
  const pulseOpacity = useRef(new Animated.Value(0)).current;
  const pulseAnim    = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    if (isRecording) {
      pulseAnim.current = Animated.loop(
        Animated.sequence([
          Animated.parallel([
            Animated.timing(pulseScale,   { toValue: 1.55, duration: PULSE_DURATION, easing: Easing.out(Easing.ease), useNativeDriver: true }),
            Animated.timing(pulseOpacity, { toValue: 0.35, duration: PULSE_DURATION / 2, useNativeDriver: true }),
          ]),
          Animated.parallel([
            Animated.timing(pulseOpacity, { toValue: 0, duration: PULSE_DURATION / 2, useNativeDriver: true }),
          ]),
        ]),
      );
      pulseAnim.current.start();
    } else {
      pulseAnim.current?.stop();
      Animated.parallel([
        Animated.timing(pulseScale,   { toValue: 1,   duration: 250, useNativeDriver: true }),
        Animated.timing(pulseOpacity, { toValue: 0,   duration: 250, useNativeDriver: true }),
      ]).start();
    }
  }, [isRecording, pulseOpacity, pulseScale]);

  // ── Inner circle scale (squish when active) ───────────────────────────────
  const innerScale = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.spring(innerScale, {
      toValue:         isRecording ? 0.78 : 1,
      useNativeDriver: true,
      bounciness:      8,
    }).start();
  }, [isRecording, innerScale]);

  // ── Press handler ─────────────────────────────────────────────────────────
  const handleMainPress = () => {
    if (state === 'idle')      return onStart();
    if (state === 'recording') return onStop();
    if (state === 'paused')    return onStop();
    if (state === 'recorded')  return onStart();   // re-record
  };

  // ── Label ─────────────────────────────────────────────────────────────────
  const label = (() => {
    if (state === 'idle')      return 'HUM';
    if (state === 'recording') return 'STOP';
    if (state === 'paused')    return 'STOP';
    if (state === 'recorded')  return 'RE-HUM';
    return 'HUM';
  })();

  const buttonColor = isActive ? Colors.accentDeep : Colors.accent;

  return (
    <View style={styles.container}>
      {/* Duration display */}
      <Text style={[styles.duration, { opacity: isActive ? 1 : 0 }]}>
        {formatDuration(durationMs)}
      </Text>

      {/* State label */}
      <Text style={styles.stateLabel}>
        {state === 'recording' ? '● REC' :
         state === 'paused'    ? '‖ PAUSED' :
         state === 'recorded'  ? '✓ READY' :
         'TAP TO RECORD'}
      </Text>

      {/* Outer ring + pulse */}
      <View style={styles.buttonWrapper}>
        {/* Pulse ring */}
        <Animated.View
          pointerEvents="none"
          style={[
            styles.pulseRing,
            {
              transform:  [{ scale: pulseScale }],
              opacity:    pulseOpacity,
              borderColor: buttonColor,
            },
          ]}
        />

        {/* Main button */}
        <Pressable
          onPress={handleMainPress}
          style={({ pressed }) => [
            styles.outerRing,
            { borderColor: buttonColor },
            pressed && styles.outerRingPressed,
          ]}
        >
          <Animated.View
            style={[
              styles.innerCircle,
              { backgroundColor: buttonColor, transform: [{ scale: innerScale }] },
            ]}
          >
            <Text style={styles.buttonLabel}>{label}</Text>
          </Animated.View>
        </Pressable>
      </View>

      {/* Pause / Resume control (only while recording) */}
      <View style={styles.secondaryRow}>
        {isRecording && (
          <TouchableOpacity onPress={onPause} style={styles.secondaryBtn}>
            <Text style={styles.secondaryBtnText}>PAUSE</Text>
          </TouchableOpacity>
        )}
        {isPaused && (
          <TouchableOpacity onPress={onResume} style={styles.secondaryBtn}>
            <Text style={styles.secondaryBtnText}>RESUME</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    alignItems:  'center',
    gap:          Spacing.lg,
  },

  duration: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXL,
    color:         Colors.inkMid,
    letterSpacing: Typography.letterSpacingWide,
    minWidth:      90,
    textAlign:     'center',
  },

  stateLabel: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeSM,
    color:         Colors.inkLight,
    letterSpacing: Typography.letterSpacingWide,
    textAlign:     'center',
  },

  buttonWrapper: {
    width:           RecordButton.size + 40,
    height:          RecordButton.size + 40,
    alignItems:      'center',
    justifyContent:  'center',
  },

  pulseRing: {
    position:    'absolute',
    width:        RecordButton.size + 20,
    height:       RecordButton.size + 20,
    borderRadius: (RecordButton.size + 20) / 2,
    borderWidth:  2,
    borderColor:  Colors.accent,
  },

  outerRing: {
    width:           RecordButton.size,
    height:          RecordButton.size,
    borderRadius:    RecordButton.size / 2,
    borderWidth:     RecordButton.borderWidth,
    borderColor:     Colors.accent,
    alignItems:      'center',
    justifyContent:  'center',
    backgroundColor: Colors.background,
    // Soft shadow
    shadowColor:     Colors.shadowDeep,
    shadowOffset:    { width: 0, height: 6 },
    shadowOpacity:   1,
    shadowRadius:    18,
    elevation:       8,
  },

  outerRingPressed: {
    opacity: 0.82,
  },

  innerCircle: {
    width:           RecordButton.innerSize,
    height:          RecordButton.innerSize,
    borderRadius:    RecordButton.innerSize / 2,
    backgroundColor: Colors.accent,
    alignItems:      'center',
    justifyContent:  'center',
  },

  buttonLabel: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeSM,
    color:         Colors.background,
    letterSpacing: Typography.letterSpacingWide,
    fontWeight:    Typography.weightBold,
  },

  secondaryRow: {
    height:         36,
    alignItems:     'center',
    justifyContent: 'center',
  },

  secondaryBtn: {
    paddingHorizontal: Spacing.lg,
    paddingVertical:   Spacing.sm,
    borderRadius:      Radii_full,
    borderWidth:       1,
    borderColor:       Colors.border,
  },

  secondaryBtnText: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeXS,
    color:         Colors.inkLight,
    letterSpacing: Typography.letterSpacingWide,
  },
});

const Radii_full = 9999;
