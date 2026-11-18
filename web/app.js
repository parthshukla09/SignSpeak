// SignSpeak web app entry point (ES module, no build step).
//
//   state.js      shared state + event bus        connection.js  WebSocket client (/ws)
//   camera.js     getUserMedia + errors           frames.js      JPEG frame sending (1 in flight)
//   overlay.js    landmarks / crop box canvas     stage.js       the one camera stage, moved per tab
//   speech.js     speechSynthesis                 settings-*.js  stored settings + drawer UI
//   translate.js  learn.js  personalise.js        the three tabs
//
// The protocol is documented in web/PROTOCOL.md.

import { api } from './js/api.js';
import { Camera } from './js/camera.js';
import { Connection } from './js/connection.js';
import { FrameSender } from './js/frames.js';
import { createLearnView } from './js/learn.js';
import { Overlay } from './js/overlay.js';
import { createPersonaliseView } from './js/personalise.js';
import { createSettingsPanel } from './js/settings-panel.js';
import { CONFIG_KEYS, applyTheme, saveSettings, settings } from './js/settings-store.js';
import { createSpeech } from './js/speech.js';
import { createStage } from './js/stage.js';
import { bus, store } from './js/state.js';
import { createTranslateView } from './js/translate.js';
import {
  announce, collect, isEditable, isKeyboardFocusedControl, toast, trackInputModality,
} from './js/util.js';

const TABS = ['translate', 'learn', 'personalise'];

function stubView(name) {
  return {
    name,
    stageSlot: () => null,
    desiredMode: () => null,
    reassertMode() {},
    enter() {},
    leave() {},
    onKey: () => false,
  };
}

/**
 * Keep the shared store in step with the server. The bus calls handlers in the order they were
 * registered, so this must run before any view subscribes: views render from store.config,
 * store.modelId and store.loadingModel inside their own msg:* handlers.
 */
function syncStoreWithServer(conn) {
  bus.on('ws:close', () => {
    store.config = null;
    store.loadingModel = null;
  });
  bus.on('msg:loading', (m) => {
    store.loadingModel = typeof m.model === 'string' && m.model ? m.model : (store.loadingModel || '…');
  });
  bus.on('msg:ready', (m) => {
    store.loadingModel = null;
    store.modelId = typeof m.model === 'string' ? m.model : null;
    store.modelLabel = typeof m.model_label === 'string' && m.model_label ? m.model_label : (store.modelId || '');
    if (m.config && typeof m.config === 'object') store.config = m.config;
    if (Array.isArray(m.classes) && m.classes.length) store.classes = m.classes;
    if (m.fresh && m.requestedModel && m.model !== m.requestedModel) {
      // The saved model isn't available any more; the server used its default.
      settings.model = null;
      saveSettings();
    }
  });
  bus.on('msg:config', (m) => {
    if (m.config && typeof m.config === 'object') store.config = m.config;
  });
  bus.on('msg:error', () => {
    if (conn.ready) store.loadingModel = null; // a failed load: the previous model stays
  });
  bus.on('msg:result', (r) => {
    store.result = r;
  });
}

function main() {
  trackInputModality();
  applyTheme();

  const stage = createStage();
  const header = collect({
    model: 'model-indicator', modelLabel: 'model-label', pill: 'status-pill', statusText: 'status-text',
    speech: 'speech-toggle', settingsBtn: 'settings-btn', banner: 'offline-banner',
    offlineText: 'offline-text', offlineRetry: 'offline-retry',
  });
  const tabs = collect({ translate: 'tab-translate', learn: 'tab-learn', personalise: 'tab-personalise' });
  const panels = collect({ translate: 'panel-translate', learn: 'panel-learn', personalise: 'panel-personalise' });
  if (!stage || !header || !tabs || !panels) {
    toast("The page didn't load correctly. Try reloading it.", { type: 'error', timeout: 60000 });
    return;
  }

  const speech = createSpeech(() => settings.speech);

  // ------------------------------------------------------------------ server connection
  function getHello() {
    let model = settings.model;
    const models = store.server && Array.isArray(store.server.models) ? store.server.models : null;
    if (model && models && !models.some((m) => m && m.id === model)) model = null; // saved model is gone
    const config = {};
    for (const key of CONFIG_KEYS) if (key in settings.config) config[key] = settings.config[key];
    return { type: 'hello', model: model || null, config };
  }

  const conn = new Connection({ bus, getHello });
  syncStoreWithServer(conn);

  /** Send a message that needs a loaded model (mode, command, learn_*, record_*). */
  function sendLive(msg, { quiet = false } = {}) {
    if (conn.live && conn.send(msg)) return true;
    if (!quiet) {
      toast(conn.isOpen ? 'The model is still loading. Try again in a moment.' : 'Not connected to the SignSpeak server.', { type: 'error' });
    }
    return false;
  }

  let stateRequest = null;
  /** GET /api/state -> store.server (shared, de-duplicated). Resolves to the state or null. */
  function refreshServerState() {
    if (stateRequest) return stateRequest;
    stateRequest = api.getState()
      .then((s) => {
        if (!s || typeof s !== 'object') return null;
        store.server = s;
        if (s.recordings && typeof s.recordings === 'object') store.recordings = s.recordings;
        if (Array.isArray(s.classes) && s.classes.length) store.classes = s.classes;
        bus.emit('server:state', s);
        renderModelChip();
        return s;
      })
      .catch((err) => {
        console.warn('[SignSpeak] /api/state failed:', err && err.message);
        return null;
      })
      .finally(() => { stateRequest = null; });
    return stateRequest;
  }

  function selectModel(id) {
    if (!id) return;
    settings.model = id;
    saveSettings();
    if (conn.switchModel(id)) {
      store.loadingModel = id;
    } else {
      toast('Not connected. Your choice is saved and loads when the server is back.');
    }
    renderModelChip();
    bus.emit('settings:changed');
  }

  // ------------------------------------------------------------------ camera, overlay, frames
  const overlay = new Overlay({
    stage: stage.els.stage,
    video: stage.els.video,
    canvas: stage.els.canvas,
    options: () => ({ mirror: settings.mirror, landmarks: settings.landmarks }),
  });

  function updateAspect() {
    const v = stage.els.video;
    if (v.videoWidth > 0 && v.videoHeight > 0) {
      document.documentElement.style.setProperty('--cam-ar', String(v.videoWidth / v.videoHeight));
    }
  }
  stage.els.video.addEventListener('loadedmetadata', updateAspect);
  stage.els.video.addEventListener('resize', updateAspect);

  const camera = new Camera(stage.els.video, (state, errorKind) => {
    store.camera = state;
    stage.setCamera(state, errorKind);
    if (state === 'live') updateAspect();
    if (state !== 'live') overlay.clear();
    renderStatus();
    frames.pump();
  });
  stage.els.retry.addEventListener('click', () => camera.start());

  const frames = new FrameSender({
    video: stage.els.video,
    conn,
    isActive: () => stage.shown && camera.live,
  });

  function applyView() {
    stage.setMirror(settings.mirror);
    overlay.request();
  }

  // ------------------------------------------------------------------ views
  const ctx = {
    bus, store, conn, speech, settings, saveSettings, stage, api,
    sendLive, selectModel, refreshServerState, applyView,
    requestStage: () => updateStage(),
  };
  const views = {
    translate: createTranslateView(ctx) || stubView('translate'),
    learn: createLearnView(ctx) || stubView('learn'),
    personalise: createPersonaliseView(ctx) || stubView('personalise'),
  };
  const settingsPanel = createSettingsPanel(ctx);

  let activeTab = null;
  let modeMismatch = 0;
  let lastModeAssert = 0;

  function updateStage() {
    const view = activeTab ? views[activeTab] : null;
    const slot = view ? view.stageSlot() : null;
    stage.place(slot);
    if (!slot) overlay.clear();
    overlay.request();
    renderStatus();
    frames.pump();
  }

  function setTab(name, { focus = false, updateHash = true } = {}) {
    const next = TABS.includes(name) ? name : 'translate';
    if (next === activeTab) return;
    const prev = activeTab;
    activeTab = next;
    modeMismatch = 0;
    lastModeAssert = performance.now();
    if (prev) views[prev].leave();
    for (const t of TABS) {
      const selected = t === next;
      tabs[t].setAttribute('aria-selected', String(selected));
      tabs[t].tabIndex = selected ? 0 : -1;
      panels[t].hidden = !selected;
    }
    stage.resetChips();
    views[next].enter();
    updateStage();
    if (focus) tabs[next].focus();
    if (updateHash) {
      try { window.history.replaceState(null, '', `#${next}`); } catch { /* ignore */ }
    }
  }

  for (const t of TABS) tabs[t].addEventListener('click', () => setTab(t));
  tabs.translate.parentElement.addEventListener('keydown', (e) => {
    const i = TABS.indexOf(activeTab);
    let j = -1;
    if (e.key === 'ArrowRight') j = (i + 1) % TABS.length;
    else if (e.key === 'ArrowLeft') j = (i - 1 + TABS.length) % TABS.length;
    else if (e.key === 'Home') j = 0;
    else if (e.key === 'End') j = TABS.length - 1;
    if (j < 0) return;
    e.preventDefault();
    setTab(TABS[j], { focus: true });
  });
  window.addEventListener('hashchange', () => {
    // Only tab names switch tabs: other fragments (the "Skip to content" link's #main) don't.
    const name = window.location.hash.slice(1);
    if (TABS.includes(name)) setTab(name, { updateHash: false });
  });

  /** If the server's mode drifts from what the view needs (e.g. after a reload), put it back. */
  function checkMode(r) {
    const view = views[activeTab];
    const desired = view && view.desiredMode();
    if (!desired || typeof r.mode !== 'string' || r.skipped || r.mode === desired) {
      modeMismatch = 0;
      return;
    }
    modeMismatch += 1;
    const now = performance.now();
    if (modeMismatch >= 3 && now - lastModeAssert > 2000) {
      modeMismatch = 0;
      lastModeAssert = now;
      view.reassertMode();
    }
  }

  // ------------------------------------------------------------------ header
  function renderModelChip() {
    const models = store.server && Array.isArray(store.server.models) ? store.server.models : [];
    const find = (id) => models.find((m) => m && m.id === id);
    let label = store.modelLabel;
    if (!label) {
      const m = find(settings.model) || find(store.server && store.server.default_model);
      label = m ? m.label : 'Model';
    }
    const loading = !!store.loadingModel;
    const text = loading ? 'Loading model…' : label;
    if (header.modelLabel.textContent !== text) header.modelLabel.textContent = text;
    header.model.classList.toggle('is-loading', loading);
    header.model.setAttribute('aria-label', `Model: ${loading ? 'loading' : label}. Open model settings`);
    header.model.title = loading ? 'Loading model…' : `${label} (click to change)`;
  }

  function renderSpeechToggle() {
    const on = speech.supported && !!settings.speech.enabled;
    header.speech.setAttribute('aria-pressed', String(on));
    header.speech.disabled = !speech.supported;
    const problem = speechProblem();
    header.speech.classList.toggle('has-problem', !!problem);
    header.speech.title = !speech.supported
      ? "Speech isn't available in this browser"
      : problem === 'blocked' ? 'Speech starts after you click the page'
        : problem === 'no-voices' ? 'No text-to-speech voice is installed'
          : on ? 'Automatic speech is on' : 'Automatic speech is off';
  }

  // ------------------------------------------------------------------ speech problems
  // Chrome/Edge only let a page speak after the user has clicked or pressed a key on it, and some
  // computers have no voices installed. Either way speech would fail silently, so say what to do.
  const speechBanner = collect({
    box: 'speech-banner', text: 'speech-banner-text', enable: 'speech-banner-enable', close: 'speech-banner-close',
  });
  const speechStatusLine = document.getElementById('set-speech-status');
  let speechBannerDismissed = '';

  function speechProblem() {
    if (!speech.supported) return null; // the settings drawer explains this case
    if (speech.status === 'no-voices') return 'no-voices';
    if (speech.status === 'blocked' || speech.needsActivation()) return 'blocked';
    if (speech.status === 'error') return 'error';
    return null;
  }

  function renderSpeechProblem() {
    const problem = speechProblem();
    if (speechBanner) {
      const show = !!problem && settings.speech.enabled && speechBannerDismissed !== problem;
      speechBanner.box.hidden = !show;
      speechBanner.enable.hidden = problem !== 'blocked';
      if (problem === 'blocked') {
        speechBanner.text.innerHTML = '<strong>Speech is waiting for a click.</strong> Your browser only lets a page '
          + 'talk after you have clicked or pressed a key on it. Click anywhere, or press <em>Enable speech</em>.';
      } else if (problem === 'no-voices') {
        speechBanner.text.innerHTML = '<strong>No text-to-speech voice is installed, so SignSpeak can\'t speak.</strong>'
          + '<small>Windows: Settings › Time &amp; language › Speech › Add voices. '
          + 'macOS: System Settings › Accessibility › Spoken Content. '
          + 'Linux: install speech-dispatcher and espeak-ng (e.g. <code>sudo apt install speech-dispatcher espeak-ng</code>) '
          + 'and restart the browser. Then reload this page.</small>';
      } else if (problem === 'error') {
        speechBanner.text.innerHTML = `<strong>Speech didn't work</strong> (${escapeHtml(speech.lastError || 'unknown error')}).`
          + '<small>Check that the sound is on and the right speaker is selected, or choose another voice in Settings. '
          + 'Online voices (for example "Google …") need an internet connection.</small>';
      }
    }
    if (speechStatusLine) {
      const n = speech.voices.length;
      speechStatusLine.textContent = !speech.supported ? ''
        : problem === 'no-voices' ? 'No voices found on this computer.'
          : problem === 'blocked' ? `${n} voice${n === 1 ? '' : 's'} available. Speech starts after your first click on the page.`
            : problem === 'error' ? `Last attempt failed: ${speech.lastError}.`
              : n ? `${n} voice${n === 1 ? '' : 's'} available.` : 'Looking for voices…';
    }
    renderSpeechToggle();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // The first click or key press anywhere makes the page "activated", which lets it speak.
  function onFirstGesture() {
    document.removeEventListener('pointerdown', onFirstGesture, true);
    document.removeEventListener('keydown', onFirstGesture, true);
    speech.unlock();
    setTimeout(renderSpeechProblem, 0);
  }
  document.addEventListener('pointerdown', onFirstGesture, true);
  document.addEventListener('keydown', onFirstGesture, true);
  speech.onStatus(renderSpeechProblem);
  speech.onVoices(renderSpeechProblem);
  if (speechBanner) {
    speechBanner.enable.addEventListener('click', () => {
      speech.unlock();
      if (settings.speech.enabled) speech.speak('Speech is on.', { interrupt: true });
      setTimeout(renderSpeechProblem, 0);
    });
    speechBanner.close.addEventListener('click', () => {
      speechBannerDismissed = speechProblem() || '';
      renderSpeechProblem();
    });
  }

  header.speech.addEventListener('click', () => {
    settings.speech.enabled = !settings.speech.enabled;
    saveSettings();
    if (!settings.speech.enabled) speech.cancel();
    renderSpeechProblem();
    bus.emit('settings:changed');
    toast(settings.speech.enabled ? 'Automatic speech is on.' : 'Automatic speech is off.');
  });
  header.settingsBtn.addEventListener('click', () => settingsPanel.open());
  header.model.addEventListener('click', () => settingsPanel.open('model'));
  header.offlineRetry.addEventListener('click', () => conn.retryNow());

  // status pill + stage status chip
  let fps = 0;
  let lastResultAt = 0;
  let lastMs = null;

  function renderStatus() {
    const s = conn.status;
    const now = performance.now();
    let text;
    let kind;
    if (s === 'offline') { text = 'Offline'; kind = 'bad'; }
    else if (s === 'connecting') { text = 'Connecting…'; kind = 'wait'; }
    else if (s === 'loading') { text = 'Loading model…'; kind = 'wait'; }
    else if (store.camera === 'error') { text = 'No camera'; kind = 'warn'; }
    else if (store.camera === 'starting') { text = 'Starting camera…'; kind = 'wait'; }
    else if (document.hidden) { text = 'Paused'; kind = 'idle'; }
    else if (!stage.shown) { text = 'Ready'; kind = 'idle'; }
    else if (fps > 0 && now - lastResultAt < 2000) { text = `Live · ${Math.round(fps)} fps`; kind = 'ok'; }
    else { text = 'Ready'; kind = 'ok'; }
    if (header.statusText.textContent !== text) header.statusText.textContent = text;
    if (header.pill.dataset.kind !== kind) header.pill.dataset.kind = kind;
    const title = kind === 'ok' && lastMs != null ? `Server time per frame: ${lastMs} ms` : '';
    if (header.pill.title !== title) header.pill.title = title;

    let stageText = null;
    if (store.camera === 'live') {
      if (s === 'offline') stageText = 'Server offline. Reconnecting…';
      else if (s === 'connecting') stageText = 'Connecting to SignSpeak…';
      else if (s === 'loading') stageText = 'Loading model…';
    }
    stage.setStatus(stageText);
  }
  setInterval(renderStatus, 500);
  document.addEventListener('visibilitychange', renderStatus);

  // offline banner (stays up from the first failure until the socket opens again)
  let bannerTimer = 0;
  function showBanner(on) {
    header.banner.hidden = !on;
    clearInterval(bannerTimer);
    bannerTimer = 0;
    if (!on) return;
    const tick = () => {
      if (conn.status === 'connecting') {
        header.offlineText.textContent = 'Connecting…';
        return;
      }
      const secs = Math.max(0, Math.ceil((conn.retryAt - Date.now()) / 1000));
      header.offlineText.textContent = secs > 0 ? `Retrying in ${secs} s.` : 'Retrying…';
    };
    tick();
    bannerTimer = setInterval(tick, 500);
  }

  // ------------------------------------------------------------------ server events
  let wasConnected = false;
  let fallbackTried = false;

  bus.on('ws:status', ({ status }) => {
    if (status === 'offline') {
      if (header.banner.hidden) showBanner(true);
      overlay.clear();
    }
    renderStatus();
  });
  bus.on('ws:open', () => {
    fallbackTried = false;
    showBanner(false);
    refreshServerState();
  });
  // The store itself is updated by syncStoreWithServer(), before these handlers run.
  bus.on('ws:close', ({ wasReady }) => {
    fps = 0;
    overlay.clear();
    renderModelChip();
    if (wasReady) {
      wasConnected = true;
      announce('Lost the connection to the SignSpeak server. Reconnecting.');
    }
  });
  bus.on('msg:loading', () => {
    renderModelChip();
    renderStatus();
  });
  bus.on('msg:ready', (m) => {
    // syncStoreWithServer() cleared settings.model if the saved model is gone
    if (m.fresh && m.requestedModel && m.model !== m.requestedModel) bus.emit('settings:changed');
    renderModelChip();
    renderStatus();
    frames.pump();
    if (m.reconnect && wasConnected) {
      toast('Reconnected to the SignSpeak server.', { type: 'success' });
      announce('Reconnected');
    }
  });
  bus.on('msg:error', (m) => {
    const message = typeof m.message === 'string' && m.message ? m.message : 'The server reported an error.';
    toast(message, { type: 'error' });
    // The first model failed to load and none is loaded: try the server default once.
    if (conn.isOpen && !conn.hasEngine && conn.status === 'loading' && !fallbackTried) {
      fallbackTried = true;
      const def = store.server && store.server.default_model;
      const fallback = def && def !== conn.helloModel ? def : (conn.helloModel !== 'original' ? 'original' : null);
      if (fallback) {
        if (settings.model) {
          settings.model = null;
          saveSettings();
        }
        conn.switchModel(fallback);
      }
    }
    renderModelChip();
  });
  bus.on('msg:result', (r) => {
    if (!r.skipped) {
      const now = performance.now();
      if (lastResultAt && now - lastResultAt < 2000) {
        const inst = 1000 / Math.max(1, now - lastResultAt);
        fps = fps ? fps * 0.85 + inst * 0.15 : inst;
      }
      lastResultAt = now;
      if (Number.isFinite(r.ms)) lastMs = r.ms;
    }
    if (stage.shown && !r.skipped) overlay.setResult(r);
    checkMode(r);
    frames.pump();
  });
  bus.on('settings:changed', () => {
    renderSpeechProblem();
    renderModelChip();
  });

  // ------------------------------------------------------------------ keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    if (document.querySelector('dialog[open]')) return;
    const target = e.target;
    if (isEditable(target)) return;
    // A control reached with the keyboard keeps Space / Enter for itself.
    if ((e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') && isKeyboardFocusedControl(target)) return;
    if (target instanceof Element && target.closest('[role="tablist"]') && String(e.key).startsWith('Arrow')) return;
    const view = views[activeTab];
    if (view && view.onKey(e)) e.preventDefault();
  });

  // ------------------------------------------------------------------ start
  applyView();
  renderModelChip();
  renderSpeechProblem();
  renderStatus();
  setTab(window.location.hash.slice(1), { updateHash: false });
  camera.start();
  refreshServerState().finally(() => conn.connect());
}

try {
  main();
} catch (err) {
  console.error('[SignSpeak] failed to start', err);
  toast('SignSpeak failed to start. Try reloading the page.', { type: 'error', timeout: 60000 });
}
