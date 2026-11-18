// Text-to-speech with the browser's speechSynthesis. Every call is guarded: some browsers have no
// speech at all, and voices load asynchronously (the 'voiceschanged' event).
//
// Why speech can be silent on some computers, and what this module reports (status):
//   'blocked'   Chrome/Edge only let a page speak after the user has clicked or pressed a key on it
//               (utterances fail with the error 'not-allowed'). The app asks for one click.
//   'no-voices' The browser exposes no voices, e.g. Linux without speech-dispatcher, or Windows
//               editions without speech packs.
//   'error'     Other synthesis failures (audio device busy, an online voice while offline, ...).
//   'unsupported' No speechSynthesis at all.  'ok' Nothing known to be wrong.

import { clamp } from './util.js';

const NO_VOICES_AFTER_MS = 2500; // voices normally arrive well within this

/** @param {() => {voice: string, rate: number}} getPrefs */
export function createSpeech(getPrefs) {
  const synth = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;
  const supported = !!synth && typeof window.SpeechSynthesisUtterance === 'function';
  let voices = [];
  let voicesChecked = false;
  let status = supported ? 'ok' : 'unsupported';
  let lastError = '';
  let lastActivity = 0;
  const live = new Set(); // keep utterances referenced until they end (Chrome may drop them otherwise)
  const voiceListeners = new Set();
  const statusListeners = new Set();

  function setStatus(next, error = '') {
    if (next === status && error === lastError) return;
    status = next;
    lastError = error;
    for (const fn of statusListeners) {
      try { fn(status, lastError); } catch (err) { console.error(err); }
    }
  }

  function loadVoices() {
    try {
      voices = (synth && synth.getVoices()) || [];
    } catch {
      voices = [];
    }
    if (voices.length && status === 'no-voices') setStatus('ok');
    for (const fn of voiceListeners) {
      try { fn(voices); } catch (err) { console.error(err); }
    }
  }

  if (supported) {
    loadVoices();
    try {
      synth.addEventListener('voiceschanged', loadVoices);
    } catch {
      synth.onvoiceschanged = loadVoices;
    }
    setTimeout(() => {
      voicesChecked = true;
      loadVoices();
      if (!voices.length) setStatus('no-voices');
    }, NO_VOICES_AFTER_MS);
  }

  /** True if this browser requires a click/key press before a page may speak and none happened yet. */
  function needsActivation() {
    const ua = typeof navigator !== 'undefined' ? navigator.userActivation : null;
    return !!ua && !ua.hasBeenActive;
  }

  function pickVoice() {
    const wanted = getPrefs().voice;
    if (!wanted) return null;
    return voices.find((v) => v.voiceURI === wanted) || null;
  }

  function utter(text, { volume = 1 } = {}) {
    const u = new SpeechSynthesisUtterance(text);
    const voice = pickVoice();
    if (voice) {
      u.voice = voice;
      u.lang = voice.lang;
    }
    u.rate = clamp(Number(getPrefs().rate) || 1, 0.5, 2);
    u.volume = volume;
    const done = () => { live.delete(u); lastActivity = Date.now(); };
    u.onstart = () => {
      lastActivity = Date.now();
      if (status === 'blocked' || status === 'error') setStatus('ok');
    };
    u.onend = done;
    u.onerror = (ev) => {
      done();
      const code = ev && ev.error;
      if (code === 'interrupted' || code === 'canceled') return; // our own cancel()
      if (code === 'not-allowed') setStatus('blocked', code);
      else setStatus('error', code || 'unknown');
      console.warn('[SignSpeak] speech error:', code);
    };
    live.add(u);
    return u;
  }

  /** Speak text. `interrupt` cancels anything queued first. Returns false if nothing was spoken. */
  function speak(text, { interrupt = false } = {}) {
    if (!supported) return false;
    const t = String(text ?? '').trim();
    if (!t) return false;
    if (needsActivation()) {
      setStatus('blocked', 'not-allowed');
      return false;
    }
    try {
      // Chrome can get stuck "speaking" forever after an audio glitch; unstick it.
      const stuck = (synth.speaking || synth.pending) && Date.now() - lastActivity > 15000;
      if (interrupt || stuck) synth.cancel();
      if (synth.paused) synth.resume();
      lastActivity = Date.now();
      synth.speak(utter(t));
      return true;
    } catch (err) {
      console.warn('[SignSpeak] speech failed', err);
      setStatus('error', String(err && err.message || err));
      return false;
    }
  }

  /** Call from a click/key handler: lets the page speak from now on and warms up the voice. */
  function unlock() {
    if (!supported) return;
    try {
      synth.speak(utter(' ', { volume: 0 }));
    } catch { /* ignore */ }
    if (status === 'blocked') setStatus('ok');
  }

  return {
    supported,
    get voices() { return voices; },
    get voicesChecked() { return voicesChecked; },
    get status() { return status; },
    get lastError() { return lastError; },
    needsActivation,
    onVoices(fn) {
      voiceListeners.add(fn);
      return () => voiceListeners.delete(fn);
    },
    onStatus(fn) {
      statusListeners.add(fn);
      return () => statusListeners.delete(fn);
    },
    speak,
    unlock,
    cancel() {
      try { synth && synth.cancel(); } catch { /* ignore */ }
    },
  };
}
