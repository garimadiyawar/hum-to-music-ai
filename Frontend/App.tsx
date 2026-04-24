import React from 'react';
import { Platform, StatusBar } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import HomeScreen       from './src/screens/HomeScreen';
import LoadingScreen    from './src/screens/LoadingScreen';
import ResultScreen     from './src/screens/ResultScreen';
import SavedFilesScreen from './src/screens/SavedFilesScreen';
import { Colors }       from './src/styles/theme';

export type RootStackParamList = {
  Home:       undefined;
  Loading: {
    recordingUri: string;          // ← local hum file passed to loading screen
  };
  Result: {
    songUrl:   string;
    humUri:    string;             // ← original hum URI now required
    duration?: number;
    key?:      string;
    tempo?:    number;
  };
  SavedFiles: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar barStyle="dark-content" backgroundColor={Colors.background} />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName="Home"
          screenOptions={{
            headerShown:       false,
            contentStyle:      { backgroundColor: Colors.background },
            animation:         'fade_from_bottom',
            animationDuration: 320,
            gestureEnabled:    true,
          }}
        >
          <Stack.Screen name="Home"       component={HomeScreen}       />
          <Stack.Screen name="Loading"    component={LoadingScreen}    options={{ gestureEnabled: false }} />
          <Stack.Screen name="Result"     component={ResultScreen}     />
          <Stack.Screen name="SavedFiles" component={SavedFilesScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}