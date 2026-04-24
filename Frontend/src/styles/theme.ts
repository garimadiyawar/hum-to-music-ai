// ─── Theme ────────────────────────────────────────────────────────────────────
// Retro analog warmth meets minimal digital clarity.
// Every colour decision references a physical material: parchment, mahogany,
// raw linen, beeswax — nothing synthetic, nothing cold.

export const Colors = {
  // Backgrounds
  background:    '#F5E6D3',   // warm cream parchment
  surface:       '#EDD9BF',   // slightly deeper linen
  surfaceRaised: '#E8CFAF',   // card / elevated surface

  // Ink & text
  ink:           '#2A1A0E',   // near-black espresso
  inkMid:        '#5C3A21',   // mahogany — primary brand colour
  inkLight:      '#8B6043',   // medium brown
  inkFaint:      '#C4A882',   // muted tan for placeholders

  // Accent
  accent:        '#5C3A21',   // mahogany
  accentDeep:    '#3B2010',   // pressed mahogany (active states)
  accentWarm:    '#8B5E3C',   // warm sienna
  highlight:     '#D7BFA6',   // soft tan highlight
  highlightDeep: '#C4A080',   // stronger highlight

  // Status
  danger:        '#9B3A2A',   // terracotta red
  success:       '#4A6741',   // muted sage

  // Borders & shadows
  border:        '#C9AB8A',
  borderLight:   '#DEC9AE',
  shadow:        'rgba(42, 26, 14, 0.15)',
  shadowDeep:    'rgba(42, 26, 14, 0.28)',
} as const;

export const Typography = {
  // Expo / React Native uses system fonts; we pick the most characterful
  // system serif on iOS (Georgia) for display and a monospaced face for labels.
  fontDisplay:  'Georgia',
  fontMono:     'Courier New',
  fontBody:     'Georgia',

  sizeXS:   11,
  sizeSM:   13,
  sizeMD:   16,
  sizeLG:   20,
  sizeXL:   28,
  sizeHero: 42,

  weightLight:   '300' as const,
  weightRegular: '400' as const,
  weightMedium:  '500' as const,
  weightBold:    '700' as const,

  letterSpacingWide:   2,
  letterSpacingNormal: 0.5,
  letterSpacingTight:  -0.5,
} as const;

export const Spacing = {
  xs:   4,
  sm:   8,
  md:   16,
  lg:   24,
  xl:   40,
  xxl:  64,
  hero: 120,
} as const;

export const Radii = {
  sm:   6,
  md:   12,
  lg:   20,
  full: 9999,
} as const;

export const Shadows = {
  soft: {
    shadowColor:   Colors.shadow,
    shadowOffset:  { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius:  8,
    elevation:     3,
  },
  deep: {
    shadowColor:   Colors.shadowDeep,
    shadowOffset:  { width: 0, height: 6 },
    shadowOpacity: 1,
    shadowRadius:  20,
    elevation:     8,
  },
} as const;

// Record button dimensions
export const RecordButton = {
  size:        120,
  innerSize:   84,
  pulseMax:    160,
  borderWidth: 3,
} as const;
