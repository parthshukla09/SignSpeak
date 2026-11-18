// Learn tab: practice setup, practice view (target letter, ring progress, streak, what the model
// sees), per-letter stats saved across sessions, and the alphabet chart.
//
// Server notes: on the frame a target is completed, learn.event is 'correct' and learn.target is
// ALREADY the next letter. learn.stats counts this session only (it restarts on learn_start), so
// the per-session counters are merged into the saved totals as deltas.

import { KEYS, load, save } from './storage.js';
import {
  LETTERS, collect, confirmDialog, h, pct, prefersReducedMotion, refUrl, setRefPhoto, toast,
} from './util.js';

const RING_CIRCUMFERENCE = 2 * Math.PI * 54; // r=54 in index.html
const WEAK_COUNT = 6;

function loadTotals() {
  const raw = load(KEYS.learnStats, {});
  const out = {};
  if (!raw || typeof raw !== 'object') return out;
  for (const L of LETTERS) {
    const s = raw[L];
    if (!s || typeof s !== 'object') continue;
    const attempts = Math.max(0, Math.floor(Number(s.attempts) || 0));
    const correct = Math.min(attempts, Math.max(0, Math.floor(Number(s.correct) || 0)));
    const best = Number(s.best_seconds);
    out[L] = { attempts, correct, best_seconds: Number.isFinite(best) && best > 0 ? best : null };
  }
  return out;
}

function level(acc) {
  if (acc >= 0.8) return 'good';
  if (acc >= 0.5) return 'mid';
  return 'low';
}

export function createLearnView(ctx) {
  const { bus, store, speech, settings, saveSettings, sendLive, requestStage } = ctx;
  const els = collect({
    setup: 'learn-setup', customWrap: 'learn-custom-wrap', customGrid: 'learn-custom-grid',
    customAll: 'learn-custom-all', customNone: 'learn-custom-none', customCount: 'learn-custom-count',
    weakWrap: 'learn-weak-wrap', weakList: 'learn-weak-list', weakNote: 'learn-weak-note',
    hold: 'learn-hold', holdVal: 'learn-hold-val', start: 'learn-start', startNote: 'learn-start-note',
    practice: 'learn-practice', slot: 'learn-stage-slot', card: 'learn-target-card',
    target: 'learn-target', ring: 'learn-ring', targetFigure: 'learn-target-figure', targetImg: 'learn-target-img',
    streak: 'learn-streak', streakVal: 'learn-streak-val', sees: 'learn-sees', seesText: 'learn-sees-text',
    holdHint: 'learn-hold-hint', sessionLetters: 'learn-session-letters',
    skip: 'learn-skip', stop: 'learn-stop', announce: 'learn-announce',
    statsBody: 'learn-stats-body', statsEmpty: 'learn-stats-empty', statsTable: 'learn-stats-table',
    statsReset: 'learn-stats-reset', chart: 'learn-chart',
  });
  if (!els) return null;

  let active = false;               // tab visible
  let phase = 'setup';              // 'setup' | 'practice'
  let params = null;                // last learn_start {letters, order, hold}
  let target = null;
  let sessionSeen = {};             // per-letter counters already merged into totals
  let awaitingStart = false;        // learn_start sent, the server's 'learn' reply not here yet
  let totals = loadTotals();
  let customSel = new Set(settings.learn.custom);
  let celebrateTimer = 0;
  const prev = { progress: -1, streak: -1, sees: '', seesState: '' };

  const scopeInputs = [...els.setup.querySelectorAll('input[name="learn-scope"]')];
  const orderInputs = [...els.setup.querySelectorAll('input[name="learn-order"]')];

  els.ring.style.strokeDasharray = String(RING_CIRCUMFERENCE);
  els.ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE);

  // ------------------------------------------------------------------ setup form
  const toggles = {};
  for (const L of LETTERS) {
    const b = h('button', { type: 'button', class: 'letter-toggle', 'aria-pressed': String(customSel.has(L)), text: L });
    b.addEventListener('click', () => {
      if (customSel.has(L)) customSel.delete(L);
      else customSel.add(L);
      settings.learn.custom = LETTERS.filter((c) => customSel.has(c));
      saveSettings();
      renderSetup();
    });
    toggles[L] = b;
    els.customGrid.append(b);
  }
  els.customAll.addEventListener('click', () => {
    customSel = new Set(LETTERS);
    settings.learn.custom = LETTERS.slice();
    saveSettings();
    renderSetup();
  });
  els.customNone.addEventListener('click', () => {
    customSel = new Set();
    settings.learn.custom = [];
    saveSettings();
    renderSetup();
  });

  for (const input of scopeInputs) {
    input.checked = input.value === settings.learn.scope;
    input.addEventListener('change', () => {
      if (!input.checked) return;
      settings.learn.scope = input.value;
      saveSettings();
      renderSetup();
    });
  }
  for (const input of orderInputs) {
    input.checked = input.value === settings.learn.order;
    input.addEventListener('change', () => {
      if (!input.checked) return;
      settings.learn.order = input.value;
      saveSettings();
    });
  }
  els.hold.value = String(settings.learn.hold);
  const holdLabel = (v) => `${Number(v).toFixed(1)} s`;
  els.holdVal.textContent = holdLabel(settings.learn.hold);
  els.hold.addEventListener('input', () => {
    els.holdVal.textContent = holdLabel(els.hold.value);
  });
  els.hold.addEventListener('change', () => {
    const v = Number(els.hold.value);
    if (Number.isFinite(v)) {
      settings.learn.hold = Math.min(3, Math.max(0.3, v));
      saveSettings();
    }
  });

  function weakLetters() {
    return LETTERS
      .filter((L) => (totals[L]?.attempts || 0) > 0)
      .map((L) => ({ L, acc: totals[L].correct / totals[L].attempts, best: totals[L].best_seconds ?? Infinity }))
      .sort((a, b) => a.acc - b.acc || b.best - a.best)
      .slice(0, WEAK_COUNT);
  }

  function selectedLetters() {
    const scope = settings.learn.scope;
    if (scope === 'custom') return LETTERS.filter((c) => customSel.has(c));
    if (scope === 'weak') return weakLetters().map((w) => w.L);
    return LETTERS.slice();
  }

  function renderSetup() {
    const scope = settings.learn.scope;
    els.customWrap.hidden = scope !== 'custom';
    els.weakWrap.hidden = scope !== 'weak';
    for (const L of LETTERS) toggles[L].setAttribute('aria-pressed', String(customSel.has(L)));
    els.customCount.textContent = `${customSel.size} selected`;

    const weak = weakLetters();
    els.weakList.replaceChildren(...weak.map((w) => h('span', { class: 'chip' }, w.L, h('small', { text: pct(w.acc) }))));
    els.weakNote.textContent = weak.length
      ? 'Your lowest-accuracy letters so far (slowest first when tied).'
      : 'Practise some letters first. Your weakest ones will be picked from your stats.';

    const letters = selectedLetters();
    els.start.disabled = letters.length === 0;
    if (scope === 'custom' && !letters.length) els.startNote.textContent = 'Choose at least one letter.';
    else if (scope === 'weak' && !letters.length) els.startNote.textContent = 'No practice stats yet.';
    else els.startNote.textContent = letters.length === 26 ? 'All 26 letters.' : `${letters.length} ${letters.length === 1 ? 'letter' : 'letters'}: ${letters.join(' ')}`;
  }

  els.setup.addEventListener('submit', (e) => {
    e.preventDefault();
    const letters = selectedLetters();
    if (!letters.length) return;
    start(letters.length === 26 ? [] : letters, settings.learn.order);
  });

  // ------------------------------------------------------------------ practice
  /**
   * Send learn_start. A frame already in flight is answered with the PREVIOUS session (its
   * target, its 'correct' event, its totals), so frame results are ignored until the server's
   * 'learn' reply, whose counters start at zero.
   */
  function sendStart(p, opts) {
    if (!sendLive({ type: 'learn_start', ...p }, opts)) return false;
    sessionSeen = {};
    awaitingStart = true;
    return true;
  }

  function start(letters, order) {
    const hold = Math.min(3, Math.max(0.3, Number(settings.learn.hold) || 0.8));
    const p = { letters, order, hold };
    if (!sendStart(p)) return false;
    params = p;
    target = null;
    phase = 'practice';
    prev.progress = -1;
    prev.streak = -1;
    prev.sees = '';
    els.target.textContent = '…';
    setRefPhoto(els.targetFigure, els.targetImg, null);
    els.holdHint.textContent = `Hold the sign for ${hold.toFixed(1)} s.`;
    els.sessionLetters.textContent = letters.length ? `Practising ${letters.join(' ')}` : 'Practising all 26 letters';
    renderPhase();
    requestStage();
    return true;
  }

  function stopPractice({ send = true } = {}) {
    if (phase !== 'practice') return;
    phase = 'setup';
    target = null;
    if (send) sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    renderPhase();
    requestStage();
  }

  function renderPhase() {
    els.setup.hidden = phase !== 'setup';
    els.practice.hidden = phase !== 'practice';
    if (phase === 'setup') renderSetup();
  }

  function setProgress(p) {
    const v = Math.max(0, Math.min(1, Number(p) || 0));
    const rounded = Math.round(v * 100);
    if (rounded === prev.progress) return;
    prev.progress = rounded;
    els.ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - v));
  }

  function setStreak(n) {
    const v = Math.max(0, Math.floor(Number(n) || 0));
    if (v === prev.streak) return;
    prev.streak = v;
    els.streakVal.textContent = String(v);
    els.streak.classList.toggle('is-zero', v === 0);
  }

  function setTarget(letter) {
    if (!letter || letter === target) return;
    target = letter;
    els.target.textContent = letter;
    setRefPhoto(els.targetFigure, els.targetImg, letter, `Reference sign for ${letter}`);
  }

  function renderSees(r) {
    const hands = Array.isArray(r.hands) && r.hands.length > 0;
    let text = 'No hands';
    let state = 'none';
    if (hands && r.best) {
      text = `${r.best} · ${pct(r.confidence)}`;
      if (r.best === target) state = r.letter ? 'match' : 'close';
      else state = 'other';
    } else if (hands) {
      text = 'Hands, no letter yet';
    }
    if (text !== prev.sees) {
      prev.sees = text;
      els.seesText.textContent = text;
    }
    if (state !== prev.seesState) {
      prev.seesState = state;
      els.sees.dataset.state = state;
    }
  }

  function celebrate(letter, next) {
    clearTimeout(celebrateTimer);
    els.card.classList.remove('is-correct');
    void els.card.offsetWidth; // restart the CSS animation (a fade when reduced motion is on)
    els.card.classList.add('is-correct');
    celebrateTimer = setTimeout(() => els.card.classList.remove('is-correct'), 900);
    els.announce.textContent = `Correct, ${letter}! ${next ? `Next: ${next}.` : ''}`;
    if (settings.speech.enabled && settings.speech.correct) speech.speak('correct', { interrupt: true });
  }

  function handleLearn(ls) {
    if (phase !== 'practice' || !ls || typeof ls !== 'object') return;
    mergeStats(ls.stats);
    if (ls.event === 'correct') {
      const done = target;
      setProgress(1);
      setTarget(ls.target);
      if (done) celebrate(done, ls.target);
      prev.progress = -1;
    } else {
      setTarget(ls.target);
      setProgress(ls.progress);
    }
    setStreak(ls.streak);
  }

  els.skip.addEventListener('click', () => {
    if (phase !== 'practice') return;
    sendLive({ type: 'learn_skip' });
  });
  els.stop.addEventListener('click', () => stopPractice());

  // ------------------------------------------------------------------ stats
  function mergeStats(stats) {
    if (!stats || typeof stats !== 'object') return;
    let changed = false;
    for (const [L, s] of Object.entries(stats)) {
      if (!LETTERS.includes(L) || !s || typeof s !== 'object') continue;
      const attempts = Math.max(0, Math.floor(Number(s.attempts) || 0));
      const correct = Math.max(0, Math.floor(Number(s.correct) || 0));
      const best = Number(s.best_seconds);
      const seen = sessionSeen[L] || { attempts: 0, correct: 0, best: null };
      const dA = Math.max(0, attempts - seen.attempts);
      const dC = Math.max(0, correct - seen.correct);
      const hasBest = Number.isFinite(best) && best > 0;
      if (dA || dC || (hasBest && best !== seen.best)) {
        const t = totals[L] || (totals[L] = { attempts: 0, correct: 0, best_seconds: null });
        t.attempts += dA;
        t.correct = Math.min(t.attempts, t.correct + dC);
        if (hasBest) t.best_seconds = t.best_seconds == null ? best : Math.min(t.best_seconds, best);
        changed = true;
      }
      sessionSeen[L] = { attempts, correct, best: hasBest ? best : seen.best };
    }
    if (changed) {
      save(KEYS.learnStats, totals);
      renderStats();
    }
  }

  function renderStats() {
    const rows = LETTERS.filter((L) => (totals[L]?.attempts || 0) > 0);
    els.statsEmpty.hidden = rows.length > 0;
    els.statsTable.hidden = rows.length === 0;
    els.statsReset.disabled = rows.length === 0;
    els.statsBody.replaceChildren(...rows.map((L) => {
      const s = totals[L];
      const acc = s.attempts ? s.correct / s.attempts : 0;
      return h('tr', null,
        h('th', { scope: 'row', text: L }),
        h('td', { text: String(s.attempts) }),
        h('td', { text: String(s.correct) }),
        h('td', null,
          h('span', { class: 'acc-cell', dataset: { level: level(acc) } },
            h('span', { class: 'meter meter--sm', 'aria-hidden': 'true' },
              h('span', { class: 'meter__fill', style: `transform: scaleX(${acc})` })),
            pct(acc))),
        h('td', { text: s.best_seconds != null ? `${s.best_seconds.toFixed(1)} s` : '–' }));
    }));
    for (const L of LETTERS) {
      const badge = chartBadges[L];
      if (!badge) continue;
      const s = totals[L];
      if (s && s.attempts) {
        const acc = s.correct / s.attempts;
        badge.textContent = pct(acc);
        badge.dataset.level = level(acc);
      } else {
        badge.textContent = 'new';
        delete badge.dataset.level;
      }
    }
    if (phase === 'setup') renderSetup();
  }

  els.statsReset.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Reset practice stats?',
      body: 'Your saved attempts, accuracy and best times for every letter will be cleared from this browser.',
      confirmText: 'Reset stats',
      danger: true,
    });
    if (!ok) return;
    totals = {};
    save(KEYS.learnStats, totals);
    // Keep counting this session from where it is, without re-adding the old counts.
    renderStats();
    toast('Practice stats reset.');
  });

  // ------------------------------------------------------------------ alphabet chart
  const chartBadges = {};
  for (const L of LETTERS) {
    const badge = h('span', { class: 'chart-item__acc', text: 'new' });
    chartBadges[L] = badge;
    const img = h('img', { src: refUrl(L), alt: '', loading: 'lazy', decoding: 'async', width: 331, height: 253 });
    img.addEventListener('error', () => { img.style.visibility = 'hidden'; });
    const btn = h('button', { type: 'button', class: 'chart-item', 'aria-label': `Practise the letter ${L}` },
      img,
      h('span', { class: 'chart-item__foot' }, h('span', { class: 'chart-item__letter', text: L }), badge));
    btn.addEventListener('click', () => {
      if (start([L], 'sequential')) {
        const behavior = prefersReducedMotion() ? 'auto' : 'smooth';
        try {
          window.scrollTo({ top: 0, behavior });
        } catch {
          window.scrollTo(0, 0);
        }
      }
    });
    els.chart.append(btn);
  }

  // ------------------------------------------------------------------ server messages
  bus.on('msg:learn', (m) => {
    awaitingStart = false;
    handleLearn(m.learn);
  });
  bus.on('msg:result', (r) => {
    if (phase !== 'practice' || !active || r.skipped) return;
    if (r.learn && !awaitingStart) handleLearn(r.learn);
    renderSees(r);
  });
  bus.on('ws:close', () => {
    awaitingStart = false; // frames in flight died with the socket
  });
  bus.on('msg:ready', (m) => {
    // A new connection starts a new server session: restart the practice there.
    if (m.reconnect && phase === 'practice' && params) {
      if (active) {
        sendStart(params, { quiet: true });
        toast('Reconnected. Practice restarted.');
      } else {
        stopPractice({ send: false });
      }
    }
  });

  renderPhase();
  renderStats();

  return {
    name: 'learn',
    stageSlot: () => (phase === 'practice' ? els.slot : null),
    desiredMode: () => (phase === 'practice' ? 'learn' : null),
    reassertMode() {
      if (phase === 'practice' && params) sendStart(params, { quiet: true });
    },
    enter() {
      active = true;
      renderPhase();
      renderStats();
    },
    leave() {
      active = false;
      // Leaving the Learn tab always switches the server back to translate mode.
      stopPractice({ send: false });
      sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    },
    onKey() {
      return false;
    },
  };
}
