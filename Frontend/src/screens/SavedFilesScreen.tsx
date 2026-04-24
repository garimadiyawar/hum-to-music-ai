import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert, FlatList, SafeAreaView, StyleSheet, Text, TouchableOpacity, View,
} from 'react-native';
import { Audio, AVPlaybackStatus } from 'expo-av';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../../App';
import { getSavedSongs, deleteSavedSong, SavedSong } from '../hooks/useSavedFiles';
import { Colors, Spacing, Typography, Shadows } from '../styles/theme';

type NavProp = NativeStackNavigationProp<RootStackParamList, 'SavedFiles'>;
interface Props { navigation: NavProp; }

export default function SavedFilesScreen({ navigation }: Props) {
  const [songs,       setSongs]       = useState<SavedSong[]>([]);
  const [playingId,   setPlayingId]   = useState<string | null>(null);
  const [sound,       setSound]       = useState<Audio.Sound | null>(null);

  const load = useCallback(async () => {
    setSongs(await getSavedSongs());
  }, []);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    return () => { sound?.unloadAsync().catch(() => {}); };
  }, [sound]);

  const handlePlay = useCallback(async (song: SavedSong) => {
    if (sound) { await sound.unloadAsync(); setSound(null); }

    if (playingId === song.id) { setPlayingId(null); return; }

    await Audio.setAudioModeAsync({ allowsRecordingIOS: false, playsInSilentModeIOS: true });
    const { sound: s } = await Audio.Sound.createAsync(
      { uri: song.songUrl },
      { shouldPlay: true },
      (status: AVPlaybackStatus) => {
        if (status.isLoaded && status.didJustFinish) setPlayingId(null);
      },
    );
    setSound(s);
    setPlayingId(song.id);
  }, [sound, playingId]);

  const handleDelete = useCallback((song: SavedSong) => {
    Alert.alert('Delete', `Remove "${song.title}"?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: async () => {
          if (playingId === song.id) { await sound?.unloadAsync(); setPlayingId(null); }
          await deleteSavedSong(song.id);
          load();
        },
      },
    ]);
  }, [playingId, sound, load]);

  const handleOpen = useCallback((song: SavedSong) => {
    sound?.unloadAsync().catch(() => {});
    setPlayingId(null);
    navigation.navigate('Result', {
      songUrl:  song.songUrl,
      humUri:   song.humUri,
      duration: song.duration,
      key:      song.key,
      tempo:    song.tempo,
    });
  }, [navigation, sound]);

  const formatDate = (ts: number) => {
    const d = new Date(ts);
    return `${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} hitSlop={16}>
          <Text style={styles.back}>← BACK</Text>
        </TouchableOpacity>
        <Text style={styles.title}>SAVED SONGS</Text>
        <View style={{ width: 50 }} />
      </View>

      {songs.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyIcon}>♪</Text>
          <Text style={styles.emptyText}>No saved songs yet.</Text>
          <Text style={styles.emptyHint}>Generate a song and tap Save.</Text>
        </View>
      ) : (
        <FlatList
          data={songs}
          keyExtractor={s => s.id}
          contentContainerStyle={styles.list}
          ItemSeparatorComponent={() => <View style={{ height: Spacing.sm }} />}
          renderItem={({ item }) => (
            <TouchableOpacity onPress={() => handleOpen(item)} style={styles.card} activeOpacity={0.8}>
              <View style={styles.cardLeft}>
                <Text style={styles.songTitle}>{item.title}</Text>
                <Text style={styles.songMeta}>
                  {[item.key, item.tempo && `${Math.round(item.tempo)} BPM`].filter(Boolean).join(' · ')}
                </Text>
                <Text style={styles.songDate}>{formatDate(item.savedAt)}</Text>
              </View>
              <View style={styles.cardRight}>
                <TouchableOpacity
                  onPress={() => handlePlay(item)}
                  style={[styles.playBtn, playingId === item.id && styles.playBtnActive]}
                >
                  <Text style={[styles.playIcon, playingId === item.id && styles.playIconActive]}>
                    {playingId === item.id ? '‖' : '▶'}
                  </Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={() => handleDelete(item)} hitSlop={12}>
                  <Text style={styles.deleteIcon}>✕</Text>
                </TouchableOpacity>
              </View>
            </TouchableOpacity>
          )}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe:  { flex: 1, backgroundColor: Colors.background },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: Spacing.lg, paddingVertical: Spacing.md,
    borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
  },
  back:  { fontFamily: Typography.fontMono, fontSize: Typography.sizeXS, color: Colors.inkLight, letterSpacing: 2 },
  title: { fontFamily: Typography.fontMono, fontSize: Typography.sizeXS, color: Colors.inkMid, letterSpacing: 3 },
  list:  { padding: Spacing.lg },
  card:  {
    backgroundColor: Colors.surface, borderRadius: 6, padding: Spacing.md,
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    borderWidth: 1, borderColor: Colors.borderLight, ...Shadows.soft,
  },
  cardLeft:  { flex: 1, gap: 4 },
  cardRight: { flexDirection: 'row', alignItems: 'center', gap: Spacing.md },
  songTitle: { fontFamily: Typography.fontDisplay, fontSize: Typography.sizeMD, color: Colors.inkMid, fontStyle: 'italic' },
  songMeta:  { fontFamily: Typography.fontMono, fontSize: Typography.sizeXS, color: Colors.inkLight, letterSpacing: 1 },
  songDate:  { fontFamily: Typography.fontMono, fontSize: 9, color: Colors.inkFaint },
  playBtn: {
    width: 40, height: 40, borderRadius: 20,
    borderWidth: 1.5, borderColor: Colors.accent,
    alignItems: 'center', justifyContent: 'center',
  },
  playBtnActive: { backgroundColor: Colors.accent },
  playIcon:      { fontSize: 14, color: Colors.accent },
  playIconActive:{ color: Colors.background },
  deleteIcon:    { fontSize: 16, color: Colors.inkFaint },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: Spacing.sm },
  emptyIcon: { fontSize: 40, color: Colors.highlight },
  emptyText: { fontFamily: Typography.fontDisplay, fontSize: Typography.sizeLG, color: Colors.inkMid, fontStyle: 'italic' },
  emptyHint: { fontFamily: Typography.fontMono, fontSize: Typography.sizeXS, color: Colors.inkFaint, letterSpacing: 2 },
});