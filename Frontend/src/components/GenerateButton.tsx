import React, { useEffect, useRef } from 'react';
import {
  Animated,
  Easing,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Colors, Spacing, Typography } from '../styles/theme';

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  onPress:   () => void;
  loading:   boolean;
  disabled?: boolean;
  progress?: number;   // 0–100 upload progress
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function GenerateButton({ onPress, loading, disabled, progress }: Props) {
  // ── Spinner animation ─────────────────────────────────────────────────────
  const spin   = useRef(new Animated.Value(0)).current;
  const spinRef = useRef<Animated.CompositeAnimation | null>(null);

  useEffect(() => {
    if (loading) {
      spin.setValue(0);
      spinRef.current = Animated.loop(
        Animated.timing(spin, {
          toValue:         1,
          duration:        1400,
          easing:          Easing.linear,
          useNativeDriver: true,
        }),
      );
      spinRef.current.start();
    } else {
      spinRef.current?.stop();
      spin.setValue(0);
    }
  }, [loading, spin]);

  const rotate = spin.interpolate({
    inputRange:  [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  // ── Entry animation ───────────────────────────────────────────────────────
  const entryOpacity = useRef(new Animated.Value(0)).current;
  const entryY       = useRef(new Animated.Value(12)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(entryOpacity, { toValue: disabled ? 0.4 : 1, duration: 320, useNativeDriver: true }),
      Animated.timing(entryY,       { toValue: 0, duration: 320, easing: Easing.out(Easing.ease), useNativeDriver: true }),
    ]).start();
  }, [disabled, entryOpacity, entryY]);

  return (
    <Animated.View style={{ opacity: entryOpacity, transform: [{ translateY: entryY }] }}>
      <TouchableOpacity
        onPress={onPress}
        disabled={disabled || loading}
        activeOpacity={0.75}
        style={[styles.button, (disabled || loading) && styles.buttonDisabled]}
      >
        {loading ? (
          <View style={styles.loadingRow}>
            <Animated.Text style={[styles.spinner, { transform: [{ rotate }] }]}>
              ◌
            </Animated.Text>
            <Text style={styles.loadingText}>
              {progress !== undefined && progress > 0
                ? `UPLOADING ${progress}%`
                : 'GENERATING…'}
            </Text>
          </View>
        ) : (
          <View style={styles.labelRow}>
            <Text style={styles.arrow}>♪</Text>
            <Text style={styles.label}>GENERATE MUSIC</Text>
            <Text style={styles.arrow}>♪</Text>
          </View>
        )}
      </TouchableOpacity>

      {/* Subtle pixel underline decoration */}
      <View style={styles.underline} />
    </Animated.View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  button: {
    backgroundColor:  Colors.accent,
    paddingVertical:   Spacing.md + 2,
    paddingHorizontal: Spacing.xl,
    borderRadius:      4,                // intentionally near-square for retro feel
    alignItems:        'center',
    justifyContent:    'center',
    shadowColor:       Colors.accentDeep,
    shadowOffset:      { width: 3, height: 5 },
    shadowOpacity:     0.55,
    shadowRadius:      0,                // hard shadow = pixel aesthetic
    elevation:         5,
  },

  buttonDisabled: {
    backgroundColor: Colors.inkLight,
    shadowOpacity:   0.1,
  },

  loadingRow: {
    flexDirection: 'row',
    alignItems:    'center',
    gap:            Spacing.sm,
  },

  spinner: {
    fontSize:  18,
    color:     Colors.background,
    lineHeight: 22,
  },

  loadingText: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeSM,
    color:         Colors.background,
    letterSpacing: Typography.letterSpacingWide,
  },

  labelRow: {
    flexDirection: 'row',
    alignItems:    'center',
    gap:            Spacing.sm,
  },

  label: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeMD,
    color:         Colors.background,
    letterSpacing: Typography.letterSpacingWide,
    fontWeight:    Typography.weightBold,
  },

  arrow: {
    fontSize: 14,
    color:    Colors.highlight,
  },

  underline: {
    height:          3,
    backgroundColor: Colors.accentDeep,
    borderBottomLeftRadius:  2,
    borderBottomRightRadius: 2,
    marginHorizontal: 3,
  },
});
