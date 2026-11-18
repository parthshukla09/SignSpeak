// Translate tab: live letter, confidence, hold-to-type, top-3, reference photo, composed text,
// editing commands, speech and history.

import { createHistory } from './history.js';
import {
  announce, collect, confirmDialog, copyText, h, pct, prefersReducedMotion, setRefPhoto, toast, truncate,
} from './util.js';

const CONFIRM_CLEAR_OVER = 12;   // characters
const POOR_RECOGNITION_MS = 6000; // uncertain this long -> show the tips again

export function createTranslateView(ctx) {
  const { bus, store, speech, settings, saveSettings, sendLive } = ctx;
  const els = collect({
    slot: 'translate-stage-slot', card: 'tr-card', glyph: 'tr-glyph', caption: 'tr-caption',
    confBar: 'tr-conf-bar', confVal: 'tr-conf-val', holdBar: 'tr-hold-bar', holdVal: 'tr-hold-val',
    release: 'tr-release-hint', top: 'tr-top', refFigure: 'tr-ref-figure', refImg: 'tr-ref-img',
    refLabel: 'tr-ref-label', tips: 'tr-tips', tipsDismiss: 'tr-tips-dismiss',
    textBox: 'text-output', textContent: 'text-content', placeholder: 'text-placeholder',
    speak: 'btn-speak', space: 'btn-space', del: 'btn-delete', clear: 'btn-clear', copy: 'btn-copy',
  });
  if (!els) return null;

  const history = createHistory({ speech });
  let active = false;
  let lastText = null;
  let uncertainSince = 0;
  let tipsAutoShown = false;
  const prev = { state: '', glyph: null, caption: null, conf: -1, hold: -1, waiting: null, top: '', ref: null, dim: null };

  // top-3 rows, built once
  const topRows = [0, 1, 2].map(() => {
    const letter = h('span', { class: 'top3__letter', text: '–' });
    const fill = h('span', { class: 'meter__fill' });
    const value = h('span', { class: 'top3__pct', text: '' });
    const row = h('li', { class: 'top3__row is-empty' }, letter, h('span', { class: 'meter meter--sm' }, fill), value);
    els.top.append(row);
    return { row, letter, fill, value };
  });

  els.tips.hidden = !!settings.tipsDismissed;

  // ------------------------------------------------------------------ rendering
  /**
   * What an edit did, for screen readers. The text box is not a live region: that would read
   * the whole sentence again after every letter.
   */
  function describeChange(before, after, word) {
    if (before == null || before === after) return null;
    if (!after) return 'Text cleared';
    if (after.startsWith(before)) {
      const added = after.slice(before.length).trim();
      if (added) return `Typed ${added}`;
      return word ? `Space. Word: ${word}` : 'Space';
    }
    if (before.startsWith(after)) {
      const removed = before.slice(after.length).trim();
      return removed ? `Deleted ${removed}` : 'Deleted space';
    }
    return null;
  }

  function setText(text, { word = null, quiet = false } = {}) {
    const t = typeof text === 'string' ? text : '';
    store.text = t;
    if (t === lastText) return;
    const change = quiet ? null : describeChange(lastText, t, word);
    lastText = t;
    if (change) announce(change);
    els.textContent.textContent = t;
    els.placeholder.hidden = t.length > 0;
    const empty = t.trim().length === 0;
    els.speak.disabled = empty;
    els.copy.disabled = empty;
    els.clear.disabled = t.length === 0;
    els.del.disabled = t.length === 0;
    els.textBox.scrollTop = els.textBox.scrollHeight;
  }

  function setBar(fill, valueEl, key, value) {
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const rounded = Math.round(v * 100);
    if (prev[key] === rounded) return;
    prev[key] = rounded;
    fill.style.transform = `scaleX(${v})`;
    valueEl.textContent = `${rounded}%`;
  }

  function renderCard(r) {
    const hands = Array.isArray(r.hands) && r.hands.length > 0;
    const state = !hands || !r.best ? 'none' : r.letter ? 'confident' : 'uncertain';
    const confidence = hands ? Number(r.confidence) || 0 : 0;

    if (state !== prev.state) {
      prev.state = state;
      els.card.dataset.state = state;
    }
    const glyph = state === 'none' ? '' : (r.letter || r.best);
    if (glyph !== prev.glyph) {
      prev.glyph = glyph;
      els.glyph.textContent = glyph;
    }
    let caption;
    if (state === 'none') caption = 'Show your hands';
    else if (state === 'uncertain') caption = `Uncertain (${r.best} ${pct(confidence)})`;
    else if (r.waiting_release) caption = `Typed ${r.letter}`;
    else caption = 'Hold steady to type';
    if (caption !== prev.caption) {
      prev.caption = caption;
      els.caption.textContent = caption;
    }

    setBar(els.confBar, els.confVal, 'conf', confidence);
    setBar(els.holdBar, els.holdVal, 'hold', state === 'confident' ? r.progress : 0);
    const waiting = !!r.waiting_release && state === 'confident';
    if (waiting !== prev.waiting) {
      prev.waiting = waiting;
      els.release.hidden = !waiting;
      els.card.classList.toggle('is-waiting', waiting);
    }

    // top-3 guesses
    const top = hands && Array.isArray(r.top) ? r.top.slice(0, 3) : [];
    const topKey = top.map((t) => `${t[0]}${Math.round((Number(t[1]) || 0) * 100)}`).join(',');
    if (topKey !== prev.top) {
      prev.top = topKey;
      topRows.forEach((row, i) => {
        const t = top[i];
        const ok = Array.isArray(t) && typeof t[0] === 'string';
        const p = ok ? Math.max(0, Math.min(1, Number(t[1]) || 0)) : 0;
        row.row.classList.toggle('is-empty', !ok);
        row.letter.textContent = ok ? t[0] : '–';
        row.value.textContent = ok ? pct(p) : '';
        row.fill.style.transform = `scaleX(${p})`;
      });
    }

    // reference photo of the letter the model sees (kept, dimmed, when hands drop)
    if (r.best && r.best !== prev.ref) {
      prev.ref = r.best;
      setRefPhoto(els.refFigure, els.refImg, r.best);
      els.refLabel.textContent = `Reference: ${r.best}`;
    }
    const dim = state === 'none';
    if (dim !== prev.dim) {
      prev.dim = dim;
      els.refFigure.classList.toggle('is-dim', dim);
    }

    // poor recognition: bring the tips back once per page load
    if (state === 'uncertain') {
      const now = performance.now();
      if (!uncertainSince) uncertainSince = now;
      if (!tipsAutoShown && els.tips.hidden && now - uncertainSince > POOR_RECOGNITION_MS) {
        tipsAutoShown = true;
        els.tips.hidden = false;
      }
    } else {
      uncertainSince = 0;
    }
  }

  function flashTyped() {
    if (prefersReducedMotion() || typeof els.glyph.animate !== 'function') return;
    try {
      els.glyph.animate(
        [{ transform: 'scale(1.18)' }, { transform: 'scale(1)' }],
        { duration: 320, easing: 'cubic-bezier(.2,.9,.3,1.3)' },
      );
    } catch { /* ignore */ }
  }

  // ------------------------------------------------------------------ speech
  function speakWord(word) {
    if (!word || !settings.speech.enabled || !settings.speech.words) return;
    speech.speak(String(word).toLowerCase());
  }

  function speakLetter(letter) {
    if (!letter || !settings.speech.enabled || !settings.speech.letters) return;
    speech.speak(letter);
  }

  // ------------------------------------------------------------------ actions
  function command(action) {
    return sendLive({ type: 'command', action });
  }

  function speakAll() {
    const text = (store.text || '').trim();
    if (!text) {
      toast('Nothing to speak yet. Sign some letters first.');
      return;
    }
    history.add(text);
    if (!speech.supported) {
      toast("Speech isn't available in this browser.", { type: 'error' });
      return;
    }
    speech.speak(text.toLowerCase(), { interrupt: true });
  }

  async function clearText() {
    const text = store.text || '';
    if (!text) return;
    if (text.trim().length > CONFIRM_CLEAR_OVER) {
      const ok = await confirmDialog({
        title: 'Clear all text?',
        body: `“${truncate(text.trim(), 80)}” will be removed. Press Speak first to keep it in your history.`,
        confirmText: 'Clear text',
        danger: true,
      });
      if (!ok) return;
    }
    command('clear');
  }

  async function copy() {
    const text = store.text || '';
    if (!text.trim()) {
      toast('Nothing to copy yet.');
      return;
    }
    const ok = await copyText(text.trim());
    toast(ok ? 'Copied to the clipboard.' : "Couldn't copy. Select the text and copy it yourself.", { type: ok ? 'success' : 'error' });
  }

  els.speak.addEventListener('click', speakAll);
  els.space.addEventListener('click', () => command('space'));
  els.del.addEventListener('click', () => command('backspace'));
  els.clear.addEventListener('click', clearText);
  els.copy.addEventListener('click', copy);
  els.tipsDismiss.addEventListener('click', () => {
    els.tips.hidden = true;
    settings.tipsDismissed = true;
    saveSettings();
  });

  // ------------------------------------------------------------------ server messages
  bus.on('msg:ready', (m) => setText(m.text, { quiet: true }));
  bus.on('msg:text', (m) => {
    setText(m.text, { word: m.word });
    if (m.word) speakWord(m.word);
  });
  bus.on('msg:result', (r) => {
    // A 'skipped' result (frame not processed) carries an empty placeholder text: ignore it.
    if (r.skipped) return;
    setText(r.text, { word: r.word });
    if (r.mode && r.mode !== 'translate') return;
    if (r.typed === ' ') {
      if (r.word) speakWord(r.word);
    } else if (typeof r.typed === 'string' && r.typed) {
      speakLetter(r.typed);
      if (active) flashTyped();
    }
    if (active) renderCard(r);
  });
  bus.on('ws:close', () => {
    if (active) renderCard({ hands: [] });
  });

  setText('', { quiet: true });
  renderCard({ hands: [] });

  return {
    name: 'translate',
    stageSlot: () => els.slot,
    desiredMode: () => 'translate',
    reassertMode() {
      sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    },
    enter() {
      active = true;
      prev.state = '';
      renderCard(store.result && (!store.result.mode || store.result.mode === 'translate') ? store.result : { hands: [] });
    },
    leave() {
      active = false;
    },
    onKey(e) {
      switch (e.key) {
        case 'Enter':
          if (!e.repeat) speakAll();
          return true;
        case ' ':
        case 'Spacebar':
          if (!e.repeat) command('space');
          return true;
        case 'Backspace':
          command('backspace');
          return true;
        case 'Escape':
          if (!e.repeat) clearText();
          return true;
        default:
          return false;
      }
    },
  };
}
