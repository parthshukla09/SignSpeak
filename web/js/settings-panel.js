// Settings drawer (<dialog id="settings-dialog">): model, recognition, view and speech.
// showModal() makes the page behind inert; Tab is additionally kept inside; Esc closes.

import {
  CONFIG_KEYS, FALLBACK_CONFIG_DEFAULTS, FALLBACK_CONFIG_LIMITS, applyTheme, resetSettings,
} from './settings-store.js';
import { collect, confirmDialog, debounce, h, lastInputWasPointer, toast, trapTab } from './util.js';

const FIELDS = [
  { key: 'threshold', step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
  { key: 'smoothing', step: 1, fmt: (v) => `${Math.round(v)} ${Math.round(v) === 1 ? 'frame' : 'frames'}` },
  { key: 'hold', step: 0.1, fmt: (v) => `${v.toFixed(1)} s` },
  { key: 'release', step: 0.05, fmt: (v) => `${v.toFixed(2)} s` },
  { key: 'auto_space', step: 0.5, fmt: (v) => (v <= 0 ? 'Off' : `${v.toFixed(1)} s`) },
];

const KIND_LABEL = { original: 'Original', personalised: 'Personalised', trained: 'Trained' };
const MODEL_KEYBOARD_DELAY_MS = 600; // arrow keys move through the radios: load only where they stop

export function createSettingsPanel(ctx) {
  const { bus, store, conn, speech, settings, saveSettings, applyView, refreshServerState, selectModel } = ctx;
  const els = collect({
    dlg: 'settings-dialog', close: 'settings-close', models: 'set-models', modelsEmpty: 'set-models-empty',
    modelsSection: 'set-models-section', flip: 'set-flip', recReset: 'set-recognition-reset',
    mirror: 'set-mirror', landmarks: 'set-landmarks', speechEnabled: 'set-speech-enabled',
    speechWords: 'set-speech-words', speechLetters: 'set-speech-letters', speechCorrect: 'set-speech-correct',
    voice: 'set-voice', rate: 'set-rate', rateVal: 'set-rate-val', testVoice: 'set-test-voice',
    speechUnsupported: 'set-speech-unsupported', reset: 'set-reset',
  });
  const sl = collect({
    threshold: 'set-threshold', thresholdVal: 'set-threshold-val',
    smoothing: 'set-smoothing', smoothingVal: 'set-smoothing-val',
    hold: 'set-hold', holdVal: 'set-hold-val',
    release: 'set-release', releaseVal: 'set-release-val',
    auto_space: 'set-auto_space', auto_spaceVal: 'set-auto_space-val',
  });
  if (!els || !sl) return { open() {}, close() {}, get isOpen() { return false; } };

  const themeInputs = [...els.dlg.querySelectorAll('input[name="set-theme"]')];
  let lastFocus = null;
  let pending = {};          // slider changes not sent yet
  let configReplies = 0;     // 'config' messages sent whose reply hasn't arrived
  let pendingModel = null;   // model chosen with the keyboard, loaded after a short pause

  const limits = () => ({ ...FALLBACK_CONFIG_LIMITS, ...((store.server && store.server.config_limits) || {}) });
  const defaults = () => ({ ...FALLBACK_CONFIG_DEFAULTS, ...((store.server && store.server.config_defaults) || {}) });
  const effective = () => ({ ...defaults(), ...settings.config, ...(store.config || {}), ...pending });

  // ------------------------------------------------------------------ recognition
  function syncSliders() {
    const lim = limits();
    const cfg = effective();
    for (const f of FIELDS) {
      const input = sl[f.key];
      const out = sl[`${f.key}Val`];
      const range = Array.isArray(lim[f.key]) && lim[f.key].length === 2 ? lim[f.key] : FALLBACK_CONFIG_LIMITS[f.key];
      input.min = String(range[0]);
      input.max = String(range[1]);
      input.step = String(f.step);
      const v = Number(cfg[f.key]);
      if (Number.isFinite(v)) {
        input.value = String(v);
        out.textContent = f.fmt(v);
      }
    }
    els.flip.checked = cfg.flip_average !== false;
  }

  function sendConfig(config) {
    // Offline: the values are saved and go out with the next 'hello'.
    if (conn.send({ type: 'config', config })) configReplies += 1;
  }

  const flushConfig = debounce(() => {
    const changes = pending;
    pending = {};
    if (Object.keys(changes).length) sendConfig(changes);
  }, 250);

  function queueConfig(key, value) {
    pending[key] = value;
    settings.config[key] = value;
    if (store.config) store.config = { ...store.config, [key]: value };
    saveSettings();
    flushConfig();
  }

  for (const f of FIELDS) {
    sl[f.key].addEventListener('input', () => {
      const raw = Number(sl[f.key].value);
      if (!Number.isFinite(raw)) return;
      const v = f.key === 'smoothing' ? Math.round(raw) : Math.round(raw * 1000) / 1000;
      sl[`${f.key}Val`].textContent = f.fmt(v);
      queueConfig(f.key, v);
    });
  }
  els.flip.addEventListener('change', () => queueConfig('flip_average', els.flip.checked));

  function sendDefaults() {
    flushConfig.cancel();
    pending = {};
    settings.config = {};
    saveSettings();
    const d = defaults();
    const cfg = {};
    for (const k of CONFIG_KEYS) if (k in d) cfg[k] = d[k];
    sendConfig(cfg);
    if (store.config) store.config = { ...store.config, ...cfg };
  }

  els.recReset.addEventListener('click', () => {
    sendDefaults();
    syncSliders();
    toast('Recognition settings are back to their defaults.');
  });

  // ------------------------------------------------------------------ models
  function renderModels() {
    const models = store.server && Array.isArray(store.server.models) ? store.server.models : [];
    const current = pendingModel || store.loadingModel || store.modelId || settings.model;
    // The list is rebuilt: put keyboard focus back on the same radio afterwards.
    const focused = els.models.contains(document.activeElement) ? document.activeElement.value : null;
    els.modelsEmpty.hidden = models.length > 0;
    els.models.replaceChildren(...models.filter((m) => m && m.id).map((m) => {
      const id = String(m.id);
      const input = h('input', { type: 'radio', name: 'set-model', value: id });
      input.checked = id === current;
      return h('label', { class: `model-option${input.checked ? ' is-checked' : ''}` },
        input,
        h('span', { class: 'model-option__body' },
          h('span', { class: 'model-option__label', text: m.label || id }),
          m.summary ? h('span', { class: 'model-option__summary', text: String(m.summary) }) : null,
          id === store.loadingModel ? h('span', { class: 'model-option__summary', text: 'Loading…' }) : null,
          h('span', { class: 'model-option__kind', dataset: { kind: m.kind || '' }, text: KIND_LABEL[m.kind] || m.kind || 'Model' })));
    }));
    if (focused != null) {
      const again = [...els.models.querySelectorAll('input[name="set-model"]')].find((i) => i.value === focused);
      if (again) again.focus({ preventScroll: true });
    }
  }

  /** Load the chosen model (each load takes seconds, so not every radio the arrows pass). */
  const commitModel = debounce(() => {
    const id = pendingModel;
    pendingModel = null;
    if (!id) return;
    if (id === (store.loadingModel || store.modelId)) {
      // Already loaded (e.g. the arrows came back to it): just remember the choice.
      settings.model = id;
      saveSettings();
      return;
    }
    selectModel(id);
  }, MODEL_KEYBOARD_DELAY_MS);

  els.models.addEventListener('change', (e) => {
    const input = e.target;
    if (!(input instanceof HTMLInputElement) || input.name !== 'set-model' || !input.checked) return;
    for (const label of els.models.querySelectorAll('.model-option')) {
      label.classList.toggle('is-checked', label.contains(input));
    }
    pendingModel = input.value;
    if (lastInputWasPointer()) commitModel.flush();
    else commitModel();
  });

  // ------------------------------------------------------------------ view
  els.mirror.addEventListener('change', () => {
    settings.mirror = els.mirror.checked;
    saveSettings();
    applyView();
  });
  els.landmarks.addEventListener('change', () => {
    settings.landmarks = els.landmarks.checked;
    saveSettings();
    applyView();
  });
  for (const input of themeInputs) {
    input.addEventListener('change', () => {
      if (!input.checked) return;
      settings.theme = input.value;
      saveSettings();
      applyTheme(settings.theme);
    });
  }

  // ------------------------------------------------------------------ speech
  function renderVoices() {
    const voices = speech.voices || [];
    const sorted = [...voices].sort((a, b) => String(a.lang).localeCompare(String(b.lang)) || String(a.name).localeCompare(String(b.name)));
    els.voice.replaceChildren(
      h('option', { value: '', text: 'System default' }),
      ...sorted.map((v) => h('option', { value: v.voiceURI, text: `${v.name} (${v.lang})${v.default ? ', default' : ''}` })),
    );
    const wanted = settings.speech.voice;
    els.voice.value = voices.some((v) => v.voiceURI === wanted) ? wanted : '';
  }
  speech.onVoices(renderVoices);

  const speechToggles = [
    [els.speechEnabled, 'enabled'],
    [els.speechWords, 'words'],
    [els.speechLetters, 'letters'],
    [els.speechCorrect, 'correct'],
  ];
  for (const [input, key] of speechToggles) {
    input.addEventListener('change', () => {
      settings.speech[key] = input.checked;
      saveSettings();
      if (key === 'enabled' && !input.checked) speech.cancel();
      bus.emit('settings:changed');
    });
  }
  els.voice.addEventListener('change', () => {
    settings.speech.voice = els.voice.value;
    saveSettings();
  });
  const rateLabel = (v) => `${Number(v).toFixed(1)}×`;
  els.rate.addEventListener('input', () => {
    const v = Number(els.rate.value);
    if (!Number.isFinite(v)) return;
    settings.speech.rate = Math.min(2, Math.max(0.5, v));
    els.rateVal.textContent = rateLabel(settings.speech.rate);
    saveSettings();
  });
  els.testVoice.addEventListener('click', () => {
    if (!speech.supported) {
      toast("Speech isn't available in this browser.", { type: 'error' });
      return;
    }
    speech.speak('Hello! This is how SignSpeak sounds.', { interrupt: true });
  });

  function syncSpeech() {
    const ok = speech.supported;
    els.speechUnsupported.hidden = ok;
    for (const [input, key] of speechToggles) {
      input.checked = !!settings.speech[key];
      input.disabled = !ok;
    }
    els.voice.disabled = !ok;
    els.rate.disabled = !ok;
    els.testVoice.disabled = !ok;
    els.rate.value = String(settings.speech.rate);
    els.rateVal.textContent = rateLabel(settings.speech.rate);
    renderVoices();
  }

  // ------------------------------------------------------------------ reset all
  els.reset.addEventListener('click', async () => {
    const ok = await confirmDialog({
      title: 'Reset settings?',
      body: 'Recognition, view and speech settings go back to their defaults. Your model choice, history, practice stats and recordings are kept.',
      confirmText: 'Reset settings',
      danger: true,
    });
    if (!ok) return;
    resetSettings();
    sendDefaults();
    applyTheme(settings.theme);
    applyView();
    syncAll();
    bus.emit('settings:changed');
    toast('Settings reset to defaults.');
  });

  // ------------------------------------------------------------------ open / close
  function syncAll() {
    syncSliders();
    renderModels();
    els.mirror.checked = !!settings.mirror;
    els.landmarks.checked = !!settings.landmarks;
    for (const input of themeInputs) input.checked = input.value === settings.theme;
    syncSpeech();
  }

  function open(section) {
    if (els.dlg.open) return;
    lastFocus = document.activeElement;
    syncAll();
    try {
      els.dlg.showModal();
    } catch {
      els.dlg.setAttribute('open', '');
    }
    let focusTarget = els.close;
    if (section === 'model') {
      focusTarget = els.models.querySelector('input:checked') || els.models.querySelector('input') || els.close;
      try { els.modelsSection.scrollIntoView({ block: 'start' }); } catch { /* ignore */ }
    }
    focusTarget.focus();
    refreshServerState();
  }

  function close() {
    if (!els.dlg.open) return;
    if (typeof els.dlg.close === 'function') els.dlg.close();
    else {
      els.dlg.removeAttribute('open');
      els.dlg.dispatchEvent(new Event('close'));
    }
  }

  els.close.addEventListener('click', close);
  els.dlg.addEventListener('close', () => {
    flushConfig.flush();
    commitModel.flush(); // a model picked with the arrows just before closing still loads
    if (lastFocus && typeof lastFocus.focus === 'function' && document.contains(lastFocus)) {
      try { lastFocus.focus({ preventScroll: true }); } catch { /* ignore */ }
    }
    lastFocus = null;
  });
  els.dlg.addEventListener('click', (e) => {
    if (e.target === els.dlg) close(); // click on the backdrop
  });
  els.dlg.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }
    trapTab(els.dlg, e);
  });

  // ------------------------------------------------------------------ live updates
  bus.on('server:state', () => {
    renderModels();
    syncSliders();
  });
  bus.on('msg:ready', () => {
    renderModels();
    syncSliders();
  });
  bus.on('msg:loading', renderModels);
  bus.on('msg:config', () => {
    // While newer changes are on their way, an older reply would move the slider back.
    configReplies = Math.max(0, configReplies - 1);
    if (configReplies === 0) syncSliders();
  });
  bus.on('ws:close', () => {
    configReplies = 0;
  });
  bus.on('settings:changed', () => {
    if (els.dlg.open) syncAll();
  });

  syncAll();

  return {
    open,
    close,
    get isOpen() { return els.dlg.open; },
  };
}
