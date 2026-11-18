// User settings, persisted in localStorage under 'signspeak.settings'.

import { KEYS, load, save } from './storage.js';

export const CONFIG_KEYS = Object.freeze(['threshold', 'smoothing', 'hold', 'release', 'auto_space', 'flip_average']);

/** Used until GET /api/state provides config_defaults / config_limits (same values as the protocol). */
export const FALLBACK_CONFIG_DEFAULTS = Object.freeze({
  threshold: 0.6, smoothing: 7, hold: 1.0, release: 0.3, auto_space: 2.0, flip_average: true,
});
export const FALLBACK_CONFIG_LIMITS = Object.freeze({
  threshold: [0, 1], smoothing: [1, 30], hold: [0, 5], release: [0, 3], auto_space: [0, 10],
});

export const DEFAULT_SETTINGS = Object.freeze({
  model: null,              // model id chosen by the user; null = server default
  config: {},               // recognition overrides the user changed (sent with 'hello')
  mirror: true,
  landmarks: true,
  theme: 'system',          // 'system' | 'dark' | 'light'
  speech: { enabled: true, words: true, letters: false, correct: true, voice: '', rate: 1 },
  learn: { scope: 'all', custom: [], order: 'random', hold: 0.8 },
  record: { takes: 3, perTake: 20, autoNext: false },
  tipsDismissed: false,
});

const clone = (v) => JSON.parse(JSON.stringify(v));
const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
const bool = (v, d) => (typeof v === 'boolean' ? v : d);
const numIn = (v, lo, hi, d) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d);
const oneOf = (v, options, d) => (options.includes(v) ? v : d);

function sanitizeConfig(raw) {
  const out = {};
  if (!isObj(raw)) return out;
  for (const key of CONFIG_KEYS) {
    const v = raw[key];
    if (key === 'flip_average') {
      if (typeof v === 'boolean') out[key] = v;
    } else if (typeof v === 'number' && Number.isFinite(v)) {
      out[key] = v;
    }
  }
  return out;
}

function sanitize(raw) {
  const d = clone(DEFAULT_SETTINGS);
  if (!isObj(raw)) return d;
  const sp = isObj(raw.speech) ? raw.speech : {};
  const le = isObj(raw.learn) ? raw.learn : {};
  const re = isObj(raw.record) ? raw.record : {};
  return {
    model: typeof raw.model === 'string' && raw.model ? raw.model : null,
    config: sanitizeConfig(raw.config),
    mirror: bool(raw.mirror, d.mirror),
    landmarks: bool(raw.landmarks, d.landmarks),
    theme: oneOf(raw.theme, ['system', 'dark', 'light'], d.theme),
    speech: {
      enabled: bool(sp.enabled, d.speech.enabled),
      words: bool(sp.words, d.speech.words),
      letters: bool(sp.letters, d.speech.letters),
      correct: bool(sp.correct, d.speech.correct),
      voice: typeof sp.voice === 'string' ? sp.voice : '',
      rate: numIn(sp.rate, 0.5, 2, d.speech.rate),
    },
    learn: {
      scope: oneOf(le.scope, ['all', 'custom', 'weak'], d.learn.scope),
      custom: Array.isArray(le.custom) ? le.custom.filter((c) => typeof c === 'string' && /^[A-Z]$/.test(c)) : [],
      order: oneOf(le.order, ['random', 'sequential'], d.learn.order),
      hold: numIn(le.hold, 0.3, 3, d.learn.hold),
    },
    record: {
      takes: Math.round(numIn(re.takes, 1, 10, d.record.takes)),
      perTake: Math.round(numIn(re.perTake, 5, 60, d.record.perTake)),
      autoNext: bool(re.autoNext, d.record.autoNext),
    },
    tipsDismissed: bool(raw.tipsDismissed, d.tipsDismissed),
  };
}

/** The live settings object. Mutate it, then call saveSettings(). */
export const settings = sanitize(load(KEYS.settings, {}));

export function saveSettings() {
  save(KEYS.settings, settings);
}

/** Reset recognition, view and speech settings. Keeps the model choice and Learn/record prefs. */
export function resetSettings() {
  const d = clone(DEFAULT_SETTINGS);
  settings.config = d.config;
  settings.mirror = d.mirror;
  settings.landmarks = d.landmarks;
  settings.theme = d.theme;
  settings.speech = d.speech;
  saveSettings();
}

/** Apply the theme: 'system' follows prefers-color-scheme, otherwise data-theme on <html>. */
export function applyTheme(theme = settings.theme) {
  const root = document.documentElement;
  if (theme === 'dark' || theme === 'light') root.setAttribute('data-theme', theme);
  else root.removeAttribute('data-theme');
}
