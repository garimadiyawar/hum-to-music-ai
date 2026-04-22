import React from 'react';
import { Animated, StyleSheet, Text } from 'react-native';
import { Colors, Spacing, Typography } from '../styles/theme';
import { Toast } from '../hooks/useToast';

interface Props {
  toast:   Toast | null;
  opacity: Animated.Value;
}

export default function ToastMessage({ toast, opacity }: Props) {
  if (!toast) return null;

  const bg =
    toast.type === 'error'   ? Colors.danger  :
    toast.type === 'success' ? Colors.success :
    Colors.accentDeep;

  return (
    <Animated.View style={[styles.container, { opacity, backgroundColor: bg }]}>
      <Text style={styles.text}>{toast.message}</Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: {
    position:          'absolute',
    bottom:             Spacing.xl + 20,
    left:               Spacing.lg,
    right:              Spacing.lg,
    paddingVertical:    Spacing.sm + 2,
    paddingHorizontal:  Spacing.md,
    borderRadius:       4,
    alignItems:         'center',
    shadowColor:        '#000',
    shadowOffset:       { width: 2, height: 3 },
    shadowOpacity:      0.3,
    shadowRadius:       0,
    elevation:          6,
    zIndex:             999,
  },
  text: {
    fontFamily:    Typography.fontMono,
    fontSize:      Typography.sizeSM,
    color:         '#fff',
    letterSpacing: Typography.letterSpacingNormal,
    textAlign:     'center',
  },
});
