// Text-to-speech with the browser's speechSynthesis. Every call is guarded: some browsers have no
// speech at all, and voices load asynchronously (the 'voiceschanged' event).

import { clamp } from './util.js';

/** @param {() => {voice: string, rate: number}} getPrefs */
export function createSpeech(getPrefs) {
  const synth = typeof window !== 'undefined' && 'speechSynthesis' in window ? window.speechSynthesis : null;
  const supported = !!synth && typeof window.SpeechSynthesisUtterance === 'function';
  let voices = [];
  const listeners = new Set();

  function loadVoices() {
    try {
      voices = (synth && synth.getVoices()) || [];
    } catch {
      voices = [];
    }
    for (const fn of listeners) {
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
  }

  function pickVoice() {
    const wanted = getPrefs().voice;
    if (!wanted) return null;
    return voices.find((v) => v.voiceURI === wanted) || null;
  }

  /** Speak text. `interrupt` cancels anything queued first. Returns false if nothing was spoken. */
  function speak(text, { interrupt = false } = {}) {
    if (!supported) return false;
    const t = String(text ?? '').trim();
    if (!t) return false;
    try {
      if (interrupt) synth.cancel();
      const u = new SpeechSynthesisUtterance(t);
      const voice = pickVoice();
      if (voice) {
        u.voice = voice;
        u.lang = voice.lang;
      }
      u.rate = clamp(Number(getPrefs().rate) || 1, 0.5, 2);
      if (synth.paused) synth.resume();
      synth.speak(u);
      return true;
    } catch (err) {
      console.warn('[SignSpeak] speech failed', err);
      return false;
    }
  }

  return {
    supported,
    get voices() { return voices; },
    onVoices(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    speak,
    cancel() {
      try { synth && synth.cancel(); } catch { /* ignore */ }
    },
  };
}
