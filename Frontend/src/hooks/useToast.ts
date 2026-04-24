import { useCallback, useRef, useState } from 'react';
import { Animated } from 'react-native';

export interface Toast {
  message: string;
  type: 'error' | 'info' | 'success';
}

export function useToast() {
  const [toast, setToast]   = useState<Toast | null>(null);
  const opacity             = useRef(new Animated.Value(0)).current;
  const timerRef            = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = useCallback((message: string, type: Toast['type'] = 'info') => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setToast({ message, type });

    Animated.sequence([
      Animated.timing(opacity, { toValue: 1, duration: 220, useNativeDriver: true }),
      Animated.delay(2800),
      Animated.timing(opacity, { toValue: 0, duration: 300, useNativeDriver: true }),
    ]).start(() => setToast(null));

    timerRef.current = setTimeout(() => setToast(null), 3400);
  }, [opacity]);

  return { toast, opacity, show };
}
