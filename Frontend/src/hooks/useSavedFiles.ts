import AsyncStorage from '@react-native-async-storage/async-storage';

export interface SavedSong {
  id:          string;
  title:       string;   // e.g. "Song 3"
  songUrl:     string;   // remote URL from server
  humUri:      string;   // local file:// URI of original hum
  key?:        string;
  tempo?:      number;
  duration?:   number;
  savedAt:     number;   // Date.now()
}

const STORAGE_KEY = 'hum_saved_songs';

export async function getSavedSongs(): Promise<SavedSong[]> {
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function saveSong(song: Omit<SavedSong, 'id' | 'title' | 'savedAt'>): Promise<void> {
  const existing = await getSavedSongs();
  const newSong: SavedSong = {
    ...song,
    id:      Date.now().toString(),
    title:   `Song ${existing.length + 1}`,
    savedAt: Date.now(),
  };
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify([newSong, ...existing]));
}

export async function deleteSavedSong(id: string): Promise<void> {
  const existing = await getSavedSongs();
  const filtered = existing.filter(s => s.id !== id);
  await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
}