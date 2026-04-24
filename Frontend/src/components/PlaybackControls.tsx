import React from 'react';
import {
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Colors, Spacing, Typography } from '../styles/theme';
import { RecorderState, formatDuration } from '../hooks/useRecorder';

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  state:        RecorderState;
  durationMs:   number;
  playbackMs:   number;
  onPlay:       () => void;
  onPause:      () => void;
  onDelete:     () => void;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function PlaybackControls({
  state,
  durationMs,
  playbackMs,
  onPlay,
  onPause,
  onDelete,
}: Props) {
  const isPlaying = state === 'playing';

  // Progress 0–1
  const progress = durationMs > 0 ? Math.min(playbackMs / durationMs, 1) : 0;

  return (
    <View style={styles.container}>
      {/* Waveform-style progress bar */}
      <ProgressBar progress={progress} />

      {/* Time row */}
      <View style={styles.timeRow}>
        <Text style={styles.timeText}>{formatDuration(playbackMs)}</Text>
        <Text style={styles.timeText}>{formatDuration(durationMs)}</Text>
      </View>

      {/* Controls row */}
      <View style={styles.controlsRow}>
        {/* Delete */}
        <TouchableOpacity onPress={onDelete} style={styles.iconBtn} hitSlop={12}>
          <Text style={styles.iconText}>✕</Text>
          <Text style={styles.iconLabel}>DELETE</Text>
        </TouchableOpacity>

        {/* Play / Pause */}
        <TouchableOpacity
          onPress={isPlaying ? onPause : onPlay}
          style={[styles.playBtn, isPlaying && styles.playBtnActive]}
        >
          <Text style={styles.playBtnIcon}>{isPlaying ? '‖' : '▶'}</Text>
        </TouchableOpacity>

        {/* Spacer (symmetric layout) */}
        <View style={styles.iconBtn} />
      </View>
    </View>
  );
}

// ─── Waveform progress bar ────────────────────────────────────────────────────

function ProgressBar({ progress }: { progress: number }) {
  const BARS     = 32;
  const filledTo = Math.round(progress * BARS);

  return (
    <View style={pbStyles.container}>
      {Array.from({ length: BARS }, (_, i) => {
        // Pseudo-random height variation for "waveform" look (deterministic)
        const seed  = ((i * 7 + 3) % 11) / 10;          // 0–1
        const h     = 6 + seed * 22;                      // 6–28 px
        const filled = i < filledTo;
        return (
          <View
            key={i}
            style={[
              pbStyles.bar,
              { height: h },
              filled ? pbStyles.barFilled : pbStyles.barEmpty,
            ]}
          />
        );
      })}
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    width:  '100%',
    gap:     Spacing.md,
    alignItems: 'center',
  },

  timeRow: {
    flexDirection:  'row',
    justifyContent: 'space-between',
    width:          '100%',
    paddingHorizontal: Spacing.xs,
  },

  timeText: {
    fontFamily: Typography.fontMono,
    fontSize:   Typography.sizeXS,
    color:      Colors.inkFaint,
  },

  controlsRow: {
    flexDirection:  'row',
    alignItems:     'center',
    justifyContent: 'space-between',
    width:          '100%',
    paddingHorizontal: Spacing.md,
  },

  iconBtn: {
    width:          52,
    alignItems:     'center',
    gap:             4,
  },

  iconText: {
    fontSize:   20,
    color:      Colors.inkLight,
  },

  iconLabel: {
    fontFamily:    Typography.fontMono,
    fontSize:      9,
    color:         Colors.inkFaint,
    letterSpacing: Typography.letterSpacingWide,
  },

  playBtn: {
    width:           64,
    height:          64,
    borderRadius:    32,
    borderWidth:     2,
    borderColor:     Colors.accent,
    alignItems:      'center',
    justifyContent:  'center',
    backgroundColor: Colors.background,
    shadowColor:     Colors.shadowDeep,
    shadowOffset:    { width: 0, height: 4 },
    shadowOpacity:   1,
    shadowRadius:    12,
    elevation:       5,
  },

  playBtnActive: {
    backgroundColor: Colors.accent,
  },

  playBtnIcon: {
    fontSize:   22,
    color:      Colors.accent,
    lineHeight: 26,
  },
});

const pbStyles = StyleSheet.create({
  container: {
    flexDirection:  'row',
    alignItems:     'flex-end',
    gap:             3,
    width:          '100%',
    height:          36,
    paddingHorizontal: 2,
  },
  bar: {
    flex:         1,
    borderRadius: 2,
  },
  barFilled: {
    backgroundColor: Colors.accent,
  },
  barEmpty: {
    backgroundColor: Colors.highlight,
  },
});
