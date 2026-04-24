import React, { useEffect, useRef, useState } from 'react';
import {
  Animated, Easing, SafeAreaView, StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../App';
import { generateMusic, getErrorMessage } from '../services/api';
import { Colors, Spacing, Typography } from '../styles/theme';

type NavProp   = NativeStackNavigationProp<RootStackParamList, 'Loading'>;
type RoutePropT = RouteProp<RootStackParamList, 'Loading'>;

interface Props { navigation: NavProp; route: RoutePropT; }

const STEPS = [
  'DETECTING PITCH…',
  'TRANSCRIBING MELODY…',
  'GENERATING HARMONY…',
  'ARRANGING INSTRUMENTS…',
  'RENDERING AUDIO…',
];

export default function LoadingScreen({ navigation, route }: Props) {
  const { recordingUri } = route.params;

  const [stepIndex,  setStepIndex]  = useState(0);
  const [uploadPct,  setUploadPct]  = useState(0);
  const [errorMsg,   setErrorMsg]   = useState<string | null>(null);

  // Spinning ring
  const spin = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.timing(spin, { toValue: 1, duration: 2000, easing: Easing.linear, useNativeDriver: true })
    ).start();
  }, [spin]);
  const rotate = spin.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  // Cycle through step labels every 2.5s to give sense of progress
  useEffect(() => {
    const id = setInterval(() => {
      setStepIndex(i => Math.min(i + 1, STEPS.length - 1));
    }, 2500);
    return () => clearInterval(id);
  }, []);

  // Fire the API call immediately on mount
  useEffect(() => {
    generateMusic(recordingUri, setUploadPct)
      .then(result => {
        navigation.replace('Result', {
          songUrl:      result.song_url,
          humUri:       recordingUri,        // ← pass original hum through
          duration:     result.duration,
          key:          result.key,
          tempo:        result.tempo,
        });
      })
      .catch(err => setErrorMsg(getErrorMessage(err)));
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.container}>

        <Text style={styles.wordmark}>HUM</Text>
        <Text style={styles.subtitle}>creating your composition</Text>

        {/* Spinning ring */}
        <View style={styles.ringWrap}>
          <Animated.View style={[styles.spinRing, { transform: [{ rotate }] }]} />
          <View style={styles.ringInner}>
            <Text style={styles.pctText}>
              {uploadPct > 0 && uploadPct < 100 ? `${uploadPct}%` : '♪'}
            </Text>
          </View>
        </View>

        {/* Step label */}
        {!errorMsg && (
          <Text style={styles.stepLabel}>{STEPS[stepIndex]}</Text>
        )}

        {/* Progress dots */}
        {!errorMsg && (
          <View style={styles.dots}>
            {STEPS.map((_, i) => (
              <View
                key={i}
                style={[styles.dot, i <= stepIndex && styles.dotActive]}
              />
            ))}
          </View>
        )}

        {/* Error state */}
        {errorMsg && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{errorMsg}</Text>
            <TouchableOpacity onPress={() => navigation.goBack()} style={styles.retryBtn}>
              <Text style={styles.retryText}>← TRY AGAIN</Text>
            </TouchableOpacity>
          </View>
        )}

      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:      { flex: 1, backgroundColor: Colors.background },
  container: {
    flex: 1, alignItems: 'center', justifyContent: 'center',
    gap: Spacing.lg, paddingHorizontal: Spacing.xl,
  },
  wordmark: {
    fontFamily: Typography.fontDisplay, fontSize: Typography.sizeHero,
    color: Colors.inkMid, letterSpacing: 18, fontStyle: 'italic',
  },
  subtitle: {
    fontFamily: Typography.fontMono, fontSize: Typography.sizeXS,
    color: Colors.inkFaint, letterSpacing: Typography.letterSpacingWide,
  },
  ringWrap: {
    width: 140, height: 140,
    alignItems: 'center', justifyContent: 'center',
    marginVertical: Spacing.lg,
  },
  spinRing: {
    position: 'absolute', width: 140, height: 140, borderRadius: 70,
    borderWidth: 2, borderColor: Colors.accent,
    borderTopColor: 'transparent',
  },
  ringInner: {
    width: 100, height: 100, borderRadius: 50,
    borderWidth: 1, borderColor: Colors.borderLight,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: Colors.surface,
  },
  pctText: {
    fontFamily: Typography.fontMono, fontSize: Typography.sizeLG,
    color: Colors.inkMid,
  },
  stepLabel: {
    fontFamily: Typography.fontMono, fontSize: Typography.sizeXS,
    color: Colors.inkLight, letterSpacing: Typography.letterSpacingWide,
  },
  dots: { flexDirection: 'row', gap: 8 },
  dot: {
    width: 6, height: 6, borderRadius: 3,
    backgroundColor: Colors.highlight,
  },
  dotActive: { backgroundColor: Colors.accent },
  errorBox: { alignItems: 'center', gap: Spacing.md, paddingTop: Spacing.lg },
  errorText: {
    fontFamily: Typography.fontMono, fontSize: Typography.sizeSM,
    color: Colors.danger, textAlign: 'center', lineHeight: 22,
  },
  retryBtn: {
    paddingVertical: Spacing.sm, paddingHorizontal: Spacing.xl,
    borderWidth: 1, borderColor: Colors.border, borderRadius: 4,
  },
  retryText: {
    fontFamily: Typography.fontMono, fontSize: Typography.sizeXS,
    color: Colors.inkMid, letterSpacing: Typography.letterSpacingWide,
  },
});