// Personalise tab: the per-letter recordings grid, the guided recording flow, and fine-tuning.
//
// Server notes (web/PROTOCOL.md + signspeak/server.py): record_start answers with a 'record'
// message whose record.active is true. After a take finishes, every later frame in record mode
// repeats that finished state (done: true) until the next record_start, so a 'done' only counts
// once the server has confirmed the NEW take with active: true.

import {
  LETTERS, announce, collect, confirmDialog, formatDuration, formatTime, h, icon, pct,
  prefersReducedMotion, setRefPhoto, toast,
} from './util.js';

const COUNTDOWN = [3, 2, 1];
const COUNTDOWN_STEP_MS = 500;      // 3 steps = 1.5 s
const SAVED_PAUSE_MS = 1300;        // pause after a saved take before moving on
const LOG_LINES = 12;
const POLL_MS = 1000;
const IDLE_STATUS = 'Press Record take (or Space) when your sign is ready.';

const clampInt = (v, lo, hi, d) => {
  const n = Math.round(Number(v));
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : d;
};

export function createPersonaliseView(ctx) {
  const { bus, store, settings, saveSettings, sendLive, requestStage, refreshServerState, api, stage, selectModel } = ctx;
  const els = collect({
    intro: 'rec-intro', card: 'rec-card', takes: 'rec-takes', perTake: 'rec-per-take',
    selMissing: 'rec-select-missing', selAll: 'rec-select-all', selNone: 'rec-select-none',
    grid: 'rec-grid', summary: 'rec-summary', start: 'rec-start', offline: 'rec-offline',
    workspace: 'rec-workspace', slot: 'rec-stage-slot', recorder: 'rec-active', step: 'rec-step',
    takeLabel: 'rec-take-label', letter: 'rec-letter', refFigure: 'rec-ref-figure', refImg: 'rec-ref-img',
    instruction: 'rec-instruction', dots: 'rec-dots', bar: 'rec-bar', barLabel: 'rec-bar-label',
    status: 'rec-status', record: 'rec-record', recordLabel: 'rec-record-label', del: 'rec-delete',
    skip: 'rec-skip', stop: 'rec-stop', auto: 'rec-auto',
    finished: 'rec-finished', finishedText: 'rec-finished-text', finishedClose: 'rec-finished-close',
    finishedFinetune: 'rec-finished-finetune',
    ftCard: 'ft-card', ftStart: 'ft-start', ftStartLabel: 'ft-start-label', ftDataset: 'ft-dataset-warning',
    ftRecheck: 'ft-recheck', ftMissing: 'ft-missing-warning', ftMissingLetters: 'ft-missing-letters',
    ftSingle: 'ft-single-warning', ftSingleLetters: 'ft-single-letters', ftNoRec: 'ft-norec-note',
    ftError: 'ft-error', ftRunning: 'ft-running', ftElapsed: 'ft-elapsed', ftLog: 'ft-log',
    ftResult: 'ft-result', ftRunName: 'ft-run-name', ftUse: 'ft-use', ftUseLabel: 'ft-use-label',
    ftHeldBefore: 'ft-heldout-before', ftHeldAfter: 'ft-heldout-after', ftHeldDelta: 'ft-heldout-delta',
    ftDataBefore: 'ft-dataset-before', ftDataAfter: 'ft-dataset-after', ftDataDelta: 'ft-dataset-delta',
    ftHeldNone: 'ft-heldout-none', ftResultMissing: 'ft-result-missing', ftLettersWrap: 'ft-letters-wrap',
    ftTable: 'ft-letters-table', ftFailed: 'ft-failed', ftFailedLog: 'ft-failed-log',
  });
  if (!els) return null;

  let active = false;
  let phase = 'overview';          // 'overview' | 'recorder' | 'finished'
  let selection = new Set();
  let selectionTouched = false;
  let queue = [];                  // letters of this recording run
  let qi = 0;                      // index of the current letter
  let goals = {};                  // letter -> takes it should have when done
  let sessionTakes = [];           // letters of the takes saved in this run, in order
  let skipped = [];
  let recordedThisRun = 0;
  let recState = 'idle';           // 'idle' | 'countdown' | 'recording' | 'saved'
  let take = null;                 // { letter, confirmed, number }
  let timers = [];

  const takesOf = (L) => Math.max(0, Math.floor(Number(store.recordings?.[L]?.takes) || 0));
  const target = () => clampInt(settings.record.takes, 1, 10, 3);
  const perTake = () => clampInt(settings.record.perTake, 5, 60, 20);
  const current = () => (phase === 'recorder' ? queue[qi] || null : null);

  // ------------------------------------------------------------------ grid
  const tiles = {};
  for (const L of LETTERS) {
    const count = h('span', { class: 'rec-tile__count', text: '–' });
    const fill = h('span', { class: 'rec-tile__fill' });
    const el = h('button', { type: 'button', class: 'rec-tile', 'aria-pressed': 'false', dataset: { letter: L } },
      h('span', { class: 'rec-tile__check', html: icon('check') }),
      h('span', { class: 'rec-tile__letter', text: L }),
      count,
      h('span', { class: 'rec-tile__bar', 'aria-hidden': 'true' }, fill));
    el.addEventListener('click', () => {
      if (phase === 'recorder') return;
      selectionTouched = true;
      if (selection.has(L)) selection.delete(L);
      else selection.add(L);
      renderGrid();
    });
    tiles[L] = { el, count, fill };
    els.grid.append(el);
  }

  function applyDefaultSelection() {
    if (selectionTouched || !store.recordings) return;
    selection = new Set(LETTERS.filter((L) => takesOf(L) < target()));
  }

  const plannedTakes = (L) => Math.max(target(), takesOf(L) + 1) - takesOf(L);

  function renderGrid() {
    const rec = store.recordings;
    const tgt = target();
    const cur = current();
    els.offline.hidden = !!rec;
    for (const L of LETTERS) {
      const t = takesOf(L);
      const tile = tiles[L];
      tile.el.dataset.status = !rec ? 'unknown' : t >= tgt ? 'complete' : t > 0 ? 'partial' : 'none';
      tile.count.textContent = rec ? `${t}/${tgt}` : '–';
      tile.fill.style.transform = `scaleX(${rec ? Math.min(1, t / tgt) : 0})`;
      tile.el.setAttribute('aria-pressed', String(selection.has(L)));
      tile.el.setAttribute('aria-label', rec ? `${L}: ${t} of ${tgt} takes recorded` : L);
      tile.el.classList.toggle('is-current', L === cur);
      tile.el.disabled = phase === 'recorder';
    }
    const letters = LETTERS.filter((L) => selection.has(L));
    const takes = letters.reduce((n, L) => n + plannedTakes(L), 0);
    const seconds = takes * (1.5 + perTake() * 0.14 + 5);
    els.summary.textContent = !rec
      ? ''
      : letters.length
        ? `${letters.length} ${letters.length === 1 ? 'letter' : 'letters'} · ${takes} ${takes === 1 ? 'take' : 'takes'} · about ${Math.max(1, Math.round(seconds / 60))} min`
        : 'Select the letters to record.';
    const busy = phase === 'recorder';
    els.start.disabled = busy || !rec || letters.length === 0;
    for (const b of [els.selMissing, els.selAll, els.selNone]) b.disabled = busy || !rec;
    els.takes.disabled = busy;
    els.perTake.disabled = busy;
  }

  els.takes.value = String(target());
  els.perTake.value = String(perTake());
  els.auto.checked = !!settings.record.autoNext;
  els.takes.addEventListener('change', () => {
    settings.record.takes = clampInt(els.takes.value, 1, 10, 3);
    els.takes.value = String(settings.record.takes);
    saveSettings();
    applyDefaultSelection();
    renderGrid();
  });
  els.perTake.addEventListener('change', () => {
    settings.record.perTake = clampInt(els.perTake.value, 5, 60, 20);
    els.perTake.value = String(settings.record.perTake);
    saveSettings();
    renderGrid();
  });
  els.auto.addEventListener('change', () => {
    settings.record.autoNext = els.auto.checked;
    saveSettings();
  });
  els.selMissing.addEventListener('click', () => {
    selectionTouched = true;
    selection = new Set(LETTERS.filter((L) => takesOf(L) < target()));
    renderGrid();
  });
  els.selAll.addEventListener('click', () => {
    selectionTouched = true;
    selection = new Set(LETTERS);
    renderGrid();
  });
  els.selNone.addEventListener('click', () => {
    selectionTouched = true;
    selection = new Set();
    renderGrid();
  });

  function setRecordings(rec) {
    if (!rec || typeof rec !== 'object') return;
    store.recordings = rec;
    applyDefaultSelection();
    render();
  }

  // ------------------------------------------------------------------ recorder
  function clearTimers() {
    timers.forEach(clearTimeout);
    timers = [];
  }

  function later(fn, ms) {
    const t = setTimeout(() => {
      timers = timers.filter((x) => x !== t);
      fn();
    }, ms);
    timers.push(t);
  }

  function setStatus(text, kind = '') {
    if (els.status.textContent !== text) els.status.textContent = text;
    els.status.dataset.kind = kind;
  }

  function renderTakeProgress(saved, needed) {
    const p = needed ? Math.min(1, saved / needed) : 0;
    els.bar.style.transform = `scaleX(${p})`;
    els.barLabel.textContent = `${saved}/${needed}`;
    if (recState === 'recording') stage.setBadge(`REC ${saved}/${needed}`, 'rec');
  }

  function startRecorder() {
    if (!store.recordings) {
      toast("Your recordings haven't loaded yet. Is the SignSpeak server running?", { type: 'error' });
      return;
    }
    const letters = LETTERS.filter((L) => selection.has(L));
    if (!letters.length) return;
    if (!sendLive({ type: 'mode', mode: 'record' })) return;
    queue = letters;
    qi = 0;
    goals = {};
    for (const L of letters) goals[L] = Math.max(target(), takesOf(L) + 1);
    sessionTakes = [];
    skipped = [];
    recordedThisRun = 0;
    recState = 'idle';
    take = null;
    phase = 'recorder';
    renderTakeProgress(0, perTake());
    setStatus(IDLE_STATUS);
    render();
    requestStage();
    try {
      els.workspace.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    } catch { /* ignore */ }
    els.record.focus({ preventScroll: true });
  }

  function recordTake() {
    if (phase !== 'recorder' || recState !== 'idle') return;
    const L = current();
    if (!L) return;
    if (store.camera !== 'live') {
      toast('The camera is not available. Fix the camera first (see the message on the video).', { type: 'error' });
      return;
    }
    if (!sendLive({ type: 'mode', mode: 'record' })) return;
    recState = 'countdown';
    setStatus('Get ready…', 'warn');
    render();
    let i = 0;
    const tick = () => {
      if (recState !== 'countdown') return;
      if (i < COUNTDOWN.length) {
        stage.setCountdown(COUNTDOWN[i]);
        i += 1;
        later(tick, COUNTDOWN_STEP_MS);
      } else {
        stage.setCountdown(null);
        beginTake(L);
      }
    };
    tick();
  }

  function beginTake(L) {
    if (!sendLive({ type: 'record_start', letter: L, per_take: perTake() })) {
      recState = 'idle';
      setStatus(IDLE_STATUS);
      render();
      return;
    }
    take = { letter: L, confirmed: false, number: null };
    recState = 'recording';
    renderTakeProgress(0, perTake());
    setStatus(`Recording ${L}. Hold the sign and move your hands a little.`, 'rec');
    render();
  }

  function handleRecord(record, recordings, handsVisible) {
    if (recordings && typeof recordings === 'object') setRecordings(recordings);
    if (phase !== 'recorder' || recState !== 'recording' || !take) return;
    if (!record || typeof record !== 'object') return;
    if (record.letter && record.letter !== take.letter) return;
    if (!take.confirmed) {
      // Ignore stale states (e.g. the previous take's 'done') until the server confirms this take.
      if (record.active === true && !record.done) {
        take.confirmed = true;
        take.number = Number.isFinite(record.take) ? record.take : null;
      } else {
        return;
      }
    }
    if (take.number != null && Number.isFinite(record.take) && record.take !== take.number) return;
    const needed = Math.max(1, Number(record.needed) || perTake());
    const saved = Math.max(0, Math.min(needed, Number(record.saved) || 0));
    renderTakeProgress(saved, needed);
    if (record.done) {
      onTakeSaved(saved, !!recordings);
      return;
    }
    if (record.active === false) {
      take = null;
      recState = 'idle';
      stage.resetChips();
      setStatus('The take stopped before it finished. Record it again.', 'warn');
      render();
      return;
    }
    if (handsVisible === false) stage.setHint('Show both hands', 'warn');
    else if (handsVisible === true) stage.setHint(null);
  }

  function onTakeSaved(saved, hadRecordings) {
    const L = take.letter;
    take = null;
    if (!hadRecordings && store.recordings) {
      const r = store.recordings[L] || { takes: 0, images: 0 };
      store.recordings[L] = { takes: (Number(r.takes) || 0) + 1, images: (Number(r.images) || 0) + saved };
    }
    sessionTakes.push(L);
    recordedThisRun += 1;
    recState = 'saved';
    stage.resetChips();
    const goal = goals[L] || target();
    const t = takesOf(L);
    setStatus(t >= goal ? `Saved. ${L} is done.` : `Saved take ${t} of ${goal} for ${L}.`, 'ok');
    announce(`Saved a take of ${L}`);
    render();
    later(advance, SAVED_PAUSE_MS);
  }

  function advance() {
    if (phase !== 'recorder' || recState !== 'saved') return;
    recState = 'idle';
    const L = current();
    if (L && takesOf(L) < (goals[L] || target())) {
      setStatus(settings.record.autoNext ? 'Next take…' : `Relax, then record the next take of ${L}. ${IDLE_STATUS}`);
      render();
      if (settings.record.autoNext && active) recordTake();
      return;
    }
    nextLetter();
  }

  function nextLetter() {
    qi += 1;
    while (qi < queue.length && takesOf(queue[qi]) >= (goals[queue[qi]] || target())) qi += 1;
    if (qi >= queue.length) {
      finish();
      return;
    }
    recState = 'idle';
    renderTakeProgress(0, perTake());
    setStatus(`Next letter: ${queue[qi]}. Compare with the reference photo, then press Record take.`);
    announce(`Next letter: ${queue[qi]}`);
    render();
  }

  /** Move past letters that are already complete (e.g. after leaving the tab mid-way). */
  function normalizePosition() {
    if (phase !== 'recorder') return;
    const L = current();
    if (L && takesOf(L) >= (goals[L] || target()) && recState === 'idle') nextLetter();
  }

  function cancelActive({ send = true } = {}) {
    clearTimers();
    stage.resetChips();
    if (recState === 'recording' && send) sendLive({ type: 'record_cancel' }, { quiet: true });
    take = null;
    recState = 'idle';
  }

  function finish() {
    cancelActive({ send: false });
    phase = 'finished';
    sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    applyDefaultSelection();
    const missing = store.recordings ? LETTERS.filter((L) => takesOf(L) === 0) : [];
    const parts = [`You recorded ${recordedThisRun} ${recordedThisRun === 1 ? 'take' : 'takes'}.`];
    if (skipped.length) parts.push(`Skipped: ${skipped.join(' ')}.`);
    parts.push(missing.length
      ? `Still not recorded: ${missing.join(' ')}. Record every letter before fine-tuning for the best results.`
      : 'Every letter has recordings, so you are ready to fine-tune.');
    els.finishedText.textContent = parts.join(' ');
    render();
    requestStage();
    els.finishedFinetune.focus({ preventScroll: false });
  }

  function skipLetter() {
    if (phase !== 'recorder') return;
    cancelActive();
    const L = current();
    if (L && takesOf(L) < (goals[L] || target()) && !skipped.includes(L)) skipped.push(L);
    nextLetter();
  }

  function stopRecorder() {
    if (phase !== 'recorder') return;
    cancelActive();
    phase = 'overview';
    sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    applyDefaultSelection();
    render();
    requestStage();
  }

  async function deleteLastTake() {
    if (phase !== 'recorder' || recState !== 'idle') return;
    const L = sessionTakes.length ? sessionTakes[sessionTakes.length - 1] : current();
    if (!L || takesOf(L) < 1) {
      toast('There is no take to delete.');
      return;
    }
    const n = takesOf(L);
    const ok = await confirmDialog({
      title: `Delete the last take of ${L}?`,
      body: `The most recent recording of ${L} is deleted from your computer. ${L} has ${n} ${n === 1 ? 'take' : 'takes'} now.`,
      confirmText: 'Delete take',
      danger: true,
    });
    if (!ok || phase !== 'recorder' || recState !== 'idle') return;
    els.del.disabled = true;
    try {
      const res = await api.deleteLastTake(L);
      if (res && res.recordings) setRecordings(res.recordings);
      else await refreshServerState();
      if (sessionTakes[sessionTakes.length - 1] === L) sessionTakes.pop();
      const idx = queue.indexOf(L);
      if (idx >= 0 && phase === 'recorder') {
        qi = idx;
        skipped = skipped.filter((x) => x !== L);
        if (takesOf(L) >= (goals[L] || target())) goals[L] = takesOf(L) + 1;
      }
      renderTakeProgress(0, perTake());
      setStatus(`Deleted the last take of ${L}. ${IDLE_STATUS}`);
      toast(`Deleted the last take of ${L}${res && res.removed ? ` (${res.removed} images)` : ''}.`, { type: 'success' });
    } catch (err) {
      toast(err && err.message ? err.message : "Couldn't delete the take.", { type: 'error' });
    } finally {
      render();
    }
  }

  function renderRecorder() {
    const L = current();
    if (!L) return;
    const goal = goals[L] || target();
    const t = takesOf(L);
    els.step.textContent = `Letter ${qi + 1} of ${queue.length}`;
    if (els.letter.textContent !== L) els.letter.textContent = L;
    setRefPhoto(els.refFigure, els.refImg, L, `Reference sign for ${L}`);
    els.instruction.textContent = `Sign ${L} your way, the way you normally would. While recording, move your hands a little: closer, further, slightly tilted.`;
    els.takeLabel.textContent = t >= goal ? `${goal} of ${goal} takes` : `Take ${t + 1} of ${goal}`;

    const n = Math.min(goal, 12);
    if (els.dots.childElementCount !== n) {
      els.dots.replaceChildren(...Array.from({ length: n }, () => h('span', { class: 'take-dot' })));
    }
    [...els.dots.children].forEach((dot, i) => {
      dot.classList.toggle('is-done', i < t);
      dot.classList.toggle('is-current', i === t && (recState === 'recording' || recState === 'countdown'));
    });

    els.record.disabled = recState !== 'idle';
    const labels = { idle: 'Record take', countdown: 'Get ready…', recording: 'Recording…', saved: 'Saved' };
    els.recordLabel.textContent = labels[recState] || 'Record take';
    const delLetter = sessionTakes.length ? sessionTakes[sessionTakes.length - 1] : L;
    els.del.disabled = recState !== 'idle' || takesOf(delLetter) < 1;
    els.del.title = takesOf(delLetter) ? `Delete the last take of ${delLetter}` : 'No take to delete';
    els.recorder.classList.toggle('is-recording', recState === 'recording');
    stage.setRecording(recState === 'recording');
  }

  function render() {
    els.intro.hidden = phase !== 'overview';
    els.workspace.hidden = phase !== 'recorder';
    els.finished.hidden = phase !== 'finished';
    renderGrid();
    if (phase === 'recorder') renderRecorder();
    renderFinetunePrecheck();
  }

  els.start.addEventListener('click', startRecorder);
  els.record.addEventListener('click', recordTake);
  els.del.addEventListener('click', deleteLastTake);
  els.skip.addEventListener('click', skipLetter);
  els.stop.addEventListener('click', stopRecorder);
  els.finishedClose.addEventListener('click', () => {
    phase = 'overview';
    render();
    els.card.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  });
  els.finishedFinetune.addEventListener('click', () => {
    phase = 'overview';
    render();
    els.ftCard.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
    if (!els.ftStart.disabled) els.ftStart.focus({ preventScroll: true });
  });

  // ------------------------------------------------------------------ fine-tuning
  let ft = null;
  let ftBusy = false;
  let ftPolling = false;
  let ftFailures = 0;
  let ftSawRunning = false;
  const tableCaption = els.ftTable.querySelector('caption');

  function letterSets() {
    if (!store.recordings) return null;
    return {
      recorded: LETTERS.filter((L) => takesOf(L) > 0),
      missing: LETTERS.filter((L) => takesOf(L) === 0),
      single: LETTERS.filter((L) => takesOf(L) === 1),
    };
  }

  /** Fine-tuning needs the bundled dataset features or the dataset (older servers only report the latter). */
  function finetuneDataMissing(s) {
    if (!s) return false;
    return 'finetune_available' in s ? s.finetune_available === false : s.dataset_available === false;
  }

  function renderFinetunePrecheck() {
    const s = store.server;
    const sets = letterSets();
    const running = ft && ft.state === 'running';
    const datasetMissing = finetuneDataMissing(s);
    els.ftDataset.hidden = !datasetMissing;
    els.ftNoRec.hidden = !sets || sets.recorded.length > 0;
    els.ftMissing.hidden = !sets || !sets.recorded.length || !sets.missing.length;
    els.ftSingle.hidden = !sets || !sets.single.length;
    if (sets) {
      els.ftMissingLetters.textContent = sets.missing.join(' ');
      els.ftSingleLetters.textContent = sets.single.join(' ');
    }
    els.ftStart.disabled = !!running || ftBusy || !s || datasetMissing || !sets || !sets.recorded.length;
    els.ftStartLabel.textContent = running ? 'Fine-tuning…' : ft && ft.state === 'done' ? 'Fine-tune again' : 'Fine-tune my model';
  }

  function showFtError(message) {
    els.ftError.hidden = !message;
    els.ftError.textContent = message || '';
  }

  function setLog(pre, text, stick) {
    if (pre.textContent === text) return;
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    pre.textContent = text;
    if (stick || atBottom) pre.scrollTop = pre.scrollHeight;
  }

  const accOf = (rep) => {
    const a = rep && typeof rep === 'object' ? Number(rep.accuracy) : NaN;
    return Number.isFinite(a) ? a : null;
  };

  function setKpi(beforeEl, afterEl, deltaEl, b, a) {
    beforeEl.textContent = b == null ? '–' : pct(b, 1);
    afterEl.textContent = a == null ? '–' : pct(a, 1);
    deltaEl.classList.remove('is-up', 'is-down');
    if (a == null || b == null) {
      deltaEl.textContent = '';
      return;
    }
    const d = (a - b) * 100;
    if (Math.abs(d) < 0.05) {
      deltaEl.textContent = 'No change';
    } else {
      deltaEl.textContent = `${d > 0 ? '+' : '−'}${Math.abs(d).toFixed(1)} percentage points`;
      deltaEl.classList.add(d > 0 ? 'is-up' : 'is-down');
    }
  }

  function renderUse() {
    const id = ft && ft.model_id;
    const inUse = !!id && store.modelId === id && !store.loadingModel;
    const loading = !!id && store.loadingModel === id;
    els.ftUse.disabled = !id || inUse || loading;
    els.ftUseLabel.textContent = inUse ? 'In use' : loading ? 'Loading…' : 'Use this model';
  }

  function renderResult(st) {
    const before = (st.result && st.result.before) || {};
    const after = (st.result && st.result.after) || {};
    const bH = before.your_heldout_take;
    const aH = after.your_heldout_take;
    setKpi(els.ftHeldBefore, els.ftHeldAfter, els.ftHeldDelta, accOf(bH), accOf(aH));
    setKpi(els.ftDataBefore, els.ftDataAfter, els.ftDataDelta, accOf(before.dataset_test), accOf(after.dataset_test));
    els.ftHeldNone.hidden = accOf(bH) != null || accOf(aH) != null;
    const finished = Number(st.finished);
    els.ftRunName.textContent = st.run
      ? `Run ${st.run}${Number.isFinite(finished) && finished > 0 ? ` · finished ${formatTime(finished * 1000)}` : ''}`
      : '';
    const missing = Array.isArray(st.missing_letters) ? st.missing_letters.filter((x) => typeof x === 'string') : [];
    // Same letters as the "Not recorded yet" warning above: don't say it twice.
    const sets = letterSets();
    const sameAsNow = !!sets && sets.recorded.length > 0 && sets.missing.join(' ') === missing.join(' ');
    els.ftResultMissing.hidden = !missing.length || sameAsNow;
    els.ftResultMissing.textContent = missing.length
      ? `These letters had no recordings when this ran: ${missing.join(' ')}. They may now be misread as letters you recorded. Record them and fine-tune again.`
      : '';

    // per-letter held-out accuracy, before -> after (letters across, rows down)
    const pb = (bH && bH.per_letter) || {};
    const pa = (aH && aH.per_letter) || {};
    const val = (o, L) => (Number.isFinite(Number(o[L])) && o[L] !== null ? Number(o[L]) : null);
    const letters = LETTERS.filter((L) => val(pb, L) != null || val(pa, L) != null);
    els.ftLettersWrap.hidden = letters.length === 0;
    const headRow = h('tr', null, h('th', { scope: 'col', text: 'Letter' }), ...letters.map((L) => h('th', { scope: 'col', text: L })));
    const beforeRow = h('tr', null, h('th', { scope: 'row', text: 'Before' }));
    const afterRow = h('tr', null, h('th', { scope: 'row', text: 'After' }));
    const changeRow = h('tr', null, h('th', { scope: 'row', text: 'Change' }));
    for (const L of letters) {
      const b = val(pb, L);
      const a = val(pa, L);
      const d = a != null && b != null ? Math.round((a - b) * 100) : null;
      const cls = d == null ? '' : d >= 5 ? 'is-up' : d <= -5 ? 'is-down' : 'is-same';
      beforeRow.append(h('td', { text: b == null ? '–' : pct(b) }));
      afterRow.append(h('td', { class: cls === 'is-same' ? '' : cls, text: a == null ? '–' : pct(a) }));
      changeRow.append(h('td', {
        class: cls,
        text: d == null ? '–' : d === 0 ? '0' : `${d > 0 ? '+' : '−'}${Math.abs(d)}`,
        title: d == null ? '' : `${L}: ${b == null ? '–' : pct(b)} → ${a == null ? '–' : pct(a)}`,
      }));
    }
    els.ftTable.replaceChildren(...[tableCaption, h('thead', null, headRow), h('tbody', null, beforeRow, afterRow, changeRow)].filter(Boolean));
    renderUse();
  }

  function renderFinetune(st) {
    ft = st && typeof st === 'object' ? st : null;
    const state = ft ? ft.state : 'idle';
    const log = ft && Array.isArray(ft.log) ? ft.log.map(String) : [];
    els.ftRunning.hidden = state !== 'running';
    els.ftResult.hidden = state !== 'done' || !ft.result;
    els.ftFailed.hidden = state !== 'failed';
    if (state === 'running') {
      ftSawRunning = true;
      setLog(els.ftLog, log.slice(-LOG_LINES).join('\n') || 'Starting…', true);
      const started = Number(ft.started);
      els.ftElapsed.textContent = Number.isFinite(started) && started > 0 ? formatDuration(Date.now() / 1000 - started) : '';
    } else if (state === 'done' && ft.result) {
      renderResult(ft);
    } else if (state === 'failed') {
      setLog(els.ftFailedLog, log.join('\n') || 'No output was captured.', true);
    }
    renderFinetunePrecheck();
  }

  function poll() {
    if (ftPolling) return;
    ftPolling = true;
    ftFailures = 0;
    const tick = async () => {
      let st = null;
      try {
        const res = await api.getFinetune();
        st = res && res.finetune;
        ftFailures = 0;
      } catch {
        ftFailures += 1;
      }
      if (st) {
        const wasRunning = ftSawRunning;
        renderFinetune(st);
        if (st.state !== 'running') {
          ftPolling = false;
          if (wasRunning) {
            ftSawRunning = false;
            if (st.state === 'done') {
              toast('Fine-tuning finished. Check the results below.', { type: 'success' });
              announce('Fine-tuning finished');
              refreshServerState();
            } else if (st.state === 'failed') {
              toast('Fine-tuning failed. The output is shown below.', { type: 'error' });
            }
          }
          return;
        }
      } else if (ftFailures === 5) {
        showFtError("Lost contact with the server. Fine-tuning may still be running; this updates when it's back.");
      }
      setTimeout(tick, POLL_MS);
    };
    tick();
  }

  async function startFinetune() {
    if (ftBusy) return;
    ftBusy = true;
    showFtError('');
    renderFinetunePrecheck();
    try {
      const s = await refreshServerState();
      if (!s) {
        showFtError("Can't reach the SignSpeak server.");
        return;
      }
      if (finetuneDataMissing(s)) {
        renderFinetunePrecheck();
        els.ftDataset.scrollIntoView({ block: 'nearest' });
        return;
      }
      const sets = letterSets();
      if (!sets || !sets.recorded.length) {
        showFtError('Record some letters first. Fine-tuning needs your recordings.');
        return;
      }
      if (sets.missing.length) {
        const ok = await confirmDialog({
          title: `${sets.missing.length} ${sets.missing.length === 1 ? 'letter has' : 'letters have'} no recordings`,
          body: `Not recorded: ${sets.missing.join(' ')}. When only some letters are recorded, the fine-tuned model can start reading the others as recorded letters (for example F read as D). Recording all 26 letters avoids this.`,
          confirmText: 'Fine-tune anyway',
          cancelText: 'Record them first',
        });
        if (!ok) {
          selectionTouched = true;
          selection = new Set(sets.missing);
          render();
          els.card.scrollIntoView({ block: 'start', behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
          return;
        }
      }
      const res = await api.startFinetune();
      renderFinetune((res && res.finetune) || { state: 'running', log: [] });
      announce('Fine-tuning started');
      poll();
    } catch (err) {
      if (err && err.status === 409) {
        toast('Fine-tuning is already running.');
        poll();
      } else {
        showFtError((err && err.message) || "Fine-tuning couldn't start.");
      }
    } finally {
      ftBusy = false;
      renderFinetunePrecheck();
    }
  }

  els.ftStart.addEventListener('click', startFinetune);
  els.ftRecheck.addEventListener('click', async () => {
    const s = await refreshServerState();
    if (s && !finetuneDataMissing(s)) toast('Fine-tuning data found. You can fine-tune now.', { type: 'success' });
    else if (s) toast('The fine-tuning data is still missing.', { type: 'error' });
  });
  els.ftUse.addEventListener('click', () => {
    const id = ft && ft.model_id;
    if (!id) return;
    selectModel(id);
    renderUse();
  });

  // ------------------------------------------------------------------ server messages
  bus.on('server:state', (s) => {
    if (s.recordings) setRecordings(s.recordings);
    if (s.finetune && typeof s.finetune === 'object') {
      if (!ftPolling) renderFinetune(s.finetune);
      if (s.finetune.state === 'running') poll();
    }
    render();
  });
  bus.on('msg:record', (m) => handleRecord(m.record, m.recordings, null));
  bus.on('msg:result', (r) => {
    if (r.skipped || (r.mode !== 'record' && !r.recordings)) return;
    handleRecord(r.record, r.recordings, Array.isArray(r.hands) ? r.hands.length > 0 : null);
  });
  bus.on('msg:ready', (m) => {
    renderUse();
    if (m.reconnect && phase === 'recorder') {
      const wasRecording = recState === 'recording' || recState === 'countdown';
      cancelActive({ send: false });
      setStatus(wasRecording ? "The connection dropped, so that take wasn't saved. Record it again." : IDLE_STATUS, wasRecording ? 'warn' : '');
      if (active) sendLive({ type: 'mode', mode: 'record' }, { quiet: true });
      render();
    }
  });
  bus.on('msg:loading', renderUse);
  bus.on('ws:close', () => {
    if (phase === 'recorder' && (recState === 'recording' || recState === 'countdown')) {
      cancelActive({ send: false });
      setStatus("The connection dropped, so that take wasn't saved.", 'warn');
      render();
    }
  });

  render();

  return {
    name: 'personalise',
    stageSlot: () => (phase === 'recorder' ? els.slot : null),
    desiredMode: () => (phase === 'recorder' ? 'record' : null),
    reassertMode() {
      if (phase !== 'recorder') return;
      if (recState === 'recording') {
        cancelActive({ send: false });
        setStatus("That take didn't finish. Record it again.", 'warn');
        render();
      }
      sendLive({ type: 'mode', mode: 'record' }, { quiet: true });
    },
    enter() {
      active = true;
      refreshServerState();
      if (phase === 'recorder') {
        sendLive({ type: 'mode', mode: 'record' }, { quiet: true });
        normalizePosition();
      }
      render();
    },
    leave() {
      active = false;
      // Leaving the tab cancels an active take and switches the server back to translate.
      if (phase === 'recorder') {
        cancelActive();
        setStatus(IDLE_STATUS);
        render();
      }
      sendLive({ type: 'mode', mode: 'translate' }, { quiet: true });
    },
    onKey(e) {
      if ((e.key === ' ' || e.key === 'Spacebar') && phase === 'recorder') {
        if (!e.repeat) recordTake();
        return true;
      }
      return false;
    },
  };
}
