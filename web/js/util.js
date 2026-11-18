// Small DOM and UI helpers shared by every module.

export const LETTERS = Object.freeze('ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''));

const reducedMotionQuery = typeof window.matchMedia === 'function'
  ? window.matchMedia('(prefers-reduced-motion: reduce)')
  : null;

export function prefersReducedMotion() {
  return !!(reducedMotionQuery && reducedMotionQuery.matches);
}

/** document.getElementById that logs (instead of throwing later) when an element is missing. */
export function byId(id) {
  const el = document.getElementById(id);
  if (!el) console.warn(`[SignSpeak] missing element #${id}`);
  return el;
}

/**
 * Look up several elements by id: collect({ name: 'element-id', ... }) -> { name: Element }.
 * Returns null (and logs every missing id) if any is missing, so a view can disable itself
 * instead of throwing on the first frame.
 */
export function collect(map) {
  const out = {};
  const missing = [];
  for (const [key, id] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (el) out[key] = el;
    else missing.push(id);
  }
  if (missing.length) {
    console.error(`[SignSpeak] missing elements: #${missing.join(', #')}`);
    return null;
  }
  return out;
}

/** Create an element: h('button', { class: 'btn', text: 'Hi', onclick: fn }, child, ...). */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'html') el.innerHTML = value; // trusted, constant markup only (icons)
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, '');
    else el.setAttribute(key, String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

/** Inline SVG icon from the sprite in index.html (constant markup, safe for innerHTML). */
export function icon(name, cls = 'icon') {
  return `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"></use></svg>`;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export function num(v, fallback = 0) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export function pct(v, digits = 0) {
  const n = Number(v);
  if (!Number.isFinite(n)) return '–';
  return `${(n * 100).toFixed(digits)}%`;
}

export function refUrl(letter) {
  return `/static/reference/${encodeURIComponent(letter)}.jpg`;
}

/** Show a reference photo in a <figure data-empty> + <img>; handles missing files. */
export function setRefPhoto(figure, img, letter, alt) {
  if (!figure || !img) return;
  if (!letter) {
    figure.dataset.empty = 'true';
    img.removeAttribute('src');
    img.dataset.letter = '';
    return;
  }
  if (img.dataset.letter === letter) return;
  img.dataset.letter = letter;
  img.alt = alt || `Reference sign for ${letter}`;
  img.onerror = () => { if (img.dataset.letter === letter) figure.dataset.empty = 'true'; };
  img.onload = () => { if (img.dataset.letter === letter) figure.dataset.empty = 'false'; };
  img.src = refUrl(letter);
  if (img.complete && img.naturalWidth) figure.dataset.empty = 'false';
}

export function debounce(fn, ms) {
  let timer = 0;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { timer = 0; fn(...args); }, ms);
  };
  wrapped.flush = (...args) => { clearTimeout(timer); timer = 0; fn(...args); };
  wrapped.cancel = () => { clearTimeout(timer); timer = 0; };
  return wrapped;
}

export function truncate(text, max) {
  const t = String(text ?? '');
  return t.length > max ? `${t.slice(0, max - 1)}…` : t;
}

export function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(num(seconds)));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}

/** True for elements where typing should go to the element, not to app shortcuts. */
export function isEditable(el) {
  if (!el || el === document.body || !(el instanceof Element)) return false;
  const tag = el.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  return !!el.isContentEditable;
}

// ---------------------------------------------------------------- input modality
// Which input moved focus last. :focus-visible can't answer "did the keyboard focus this
// control?": Chrome turns it on for a mouse-focused button as soon as any key is pressed,
// before the keydown handlers run, so it would always say yes for the key being handled.
const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'AltGraph', 'CapsLock', 'Fn', 'OS']);
let lastInput = 'pointer';
let keyboardFocused = null;   // the element that last received focus from the keyboard

/** Start tracking (call once at startup, before other listeners). */
export function trackInputModality() {
  document.addEventListener('keydown', (e) => {
    if (!MODIFIER_KEYS.has(e.key)) lastInput = 'keyboard';
  }, true);
  document.addEventListener('pointerdown', () => {
    lastInput = 'pointer';
    keyboardFocused = null; // clicking even the focused control means "now using the pointer"
  }, true);
  document.addEventListener('focusin', (e) => {
    keyboardFocused = lastInput === 'keyboard' ? e.target : null;
  }, true);
}

/** True when the last interaction was a pointer (mouse, touch, pen) rather than a key. */
export function lastInputWasPointer() {
  return lastInput === 'pointer';
}

/** A control the user reached with the keyboard: Space/Enter must activate it, not shortcuts. */
export function isKeyboardFocusedControl(el) {
  if (!el || el === document.body || !(el instanceof Element)) return false;
  if (!el.matches('button, a[href], summary, [role="button"], [role="tab"], [role="switch"]')) return false;
  return el === keyboardFocused;
}

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/** Keep Tab / Shift+Tab inside `container` (used by dialogs on top of showModal's inertness). */
export function trapTab(container, event) {
  if (event.key !== 'Tab') return;
  const items = [...container.querySelectorAll(FOCUSABLE)]
    .filter((el) => el.getClientRects().length > 0 && !el.closest('[hidden]'));
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && (active === first || !container.contains(active))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (active === last || !container.contains(active))) {
    event.preventDefault();
    first.focus();
  }
}

// ---------------------------------------------------------------- screen reader announcements
let srTimer = 0;
export function announce(text) {
  const el = document.getElementById('sr-status');
  if (!el) return;
  el.textContent = '';
  clearTimeout(srTimer);
  srTimer = setTimeout(() => { el.textContent = text; }, 60);
}

// ---------------------------------------------------------------- toasts
const MAX_TOASTS = 4;
const recentToasts = new Map();

export function toast(message, { type = 'info', timeout } = {}) {
  const host = document.getElementById('toasts');
  const text = String(message ?? '').trim();
  if (!host || !text) return;
  // The same message twice within 2.5 s is noise (e.g. repeated server errors).
  const now = Date.now();
  if (now - (recentToasts.get(text) || 0) < 2500) return;
  recentToasts.set(text, now);
  if (recentToasts.size > 50) recentToasts.clear();

  const iconName = type === 'error' ? 'alert' : type === 'success' ? 'check' : 'info';
  const el = h('div', { class: `toast toast--${type}` });
  el.innerHTML = icon(iconName);
  el.append(h('span', { class: 'toast__msg', text }));
  const close = h('button', { class: 'toast__close', type: 'button', 'aria-label': 'Dismiss notification', html: icon('x') });
  el.append(close);

  let removed = false;
  const remove = () => {
    if (removed) return;
    removed = true;
    el.classList.add('is-leaving');
    setTimeout(() => el.remove(), prefersReducedMotion() ? 0 : 200);
  };
  close.addEventListener('click', remove);
  host.append(el);
  while (host.children.length > MAX_TOASTS) host.firstElementChild?.remove();
  setTimeout(remove, timeout ?? (type === 'error' ? 7000 : 3800));
}

// ---------------------------------------------------------------- confirm dialog
/** Promise<boolean>. Uses the <dialog id="confirm-dialog"> in index.html. */
export function confirmDialog({ title, body = '', confirmText = 'OK', cancelText = 'Cancel', danger = false }) {
  const els = collect({
    dlg: 'confirm-dialog', title: 'confirm-title', body: 'confirm-body',
    ok: 'confirm-ok', cancel: 'confirm-cancel',
  });
  if (!els || typeof els.dlg.showModal !== 'function') {
    return Promise.resolve(window.confirm(body ? `${title}\n\n${body}` : title));
  }
  const { dlg } = els;
  if (dlg.open) return Promise.resolve(false);

  return new Promise((resolve) => {
    els.title.textContent = title;
    els.body.textContent = body;
    els.body.hidden = !body;
    els.ok.textContent = confirmText;
    els.cancel.textContent = cancelText;
    els.ok.className = danger ? 'btn btn--danger-solid' : 'btn btn--primary';
    const previous = document.activeElement;
    let result = false;

    const onOk = () => { result = true; dlg.close(); };
    const onCancel = () => { dlg.close(); };
    const onKey = (e) => trapTab(dlg, e);
    const onBackdrop = (e) => { if (e.target === dlg) dlg.close(); };
    const onClose = () => {
      els.ok.removeEventListener('click', onOk);
      els.cancel.removeEventListener('click', onCancel);
      dlg.removeEventListener('keydown', onKey);
      dlg.removeEventListener('click', onBackdrop);
      if (previous && typeof previous.focus === 'function' && document.contains(previous)) {
        try { previous.focus({ preventScroll: true }); } catch { /* ignore */ }
      }
      resolve(result);
    };
    els.ok.addEventListener('click', onOk);
    els.cancel.addEventListener('click', onCancel);
    dlg.addEventListener('keydown', onKey);
    dlg.addEventListener('click', onBackdrop);
    dlg.addEventListener('close', onClose, { once: true });
    try {
      dlg.showModal();
    } catch {
      onClose();
      return;
    }
    (danger ? els.cancel : els.ok).focus();
  });
}

// ---------------------------------------------------------------- clipboard
export async function copyText(text) {
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch { /* fall back below */ }
  try {
    const ta = h('textarea', { readonly: true, 'aria-hidden': 'true', style: 'position:fixed;top:0;left:0;opacity:0' });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export function formatTime(ts) {
  try {
    const d = new Date(ts);
    const today = new Date();
    const sameDay = d.toDateString() === today.toDateString();
    return sameDay
      ? d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      : d.toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}
