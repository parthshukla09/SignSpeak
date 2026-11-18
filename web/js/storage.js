// localStorage access. Every call is wrapped: storage can be disabled, full or throw in private
// windows, and the app must keep working without it.

export const KEYS = Object.freeze({
  settings: 'signspeak.settings',
  history: 'signspeak.history',
  learnStats: 'signspeak.learnStats',
});

export function load(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw == null) return fallback;
    const value = JSON.parse(raw);
    return value ?? fallback;
  } catch {
    return fallback;
  }
}

export function save(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function remove(key) {
  try {
    window.localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}
