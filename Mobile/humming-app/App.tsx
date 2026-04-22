import React from 'react';
import { Platform, StatusBar } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import HomeScreen   from './src/screens/HomeScreen';
import ResultScreen from './src/screens/ResultScreen';
import { Colors }   from './src/styles/theme';

// ─── Navigation types ─────────────────────────────────────────────────────────

export type RootStackParamList = {
  Home:   undefined;
  Result: {
    songUrl:  string;
    duration?: number;
    key?:      string;
    tempo?:    number;
  };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

// ─── Root component ───────────────────────────────────────────────────────────

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar barStyle="dark-content" backgroundColor={Colors.background} />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerShown:         false,
            contentStyle:        { backgroundColor: Colors.background },
            animation:           'fade_from_bottom',
            animationDuration:   320,
            gestureEnabled:      true,
            gestureDirection:    'vertical',
          }}
        >
          <Stack.Screen name="Home"   component={HomeScreen}   />
          <Stack.Screen name="Result" component={ResultScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}
