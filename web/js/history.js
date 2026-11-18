// Spoken-text history (localStorage 'signspeak.history', newest first, at most 50 entries).

import { KEYS, load, save } from './storage.js';
import { collect, confirmDialog, copyText, formatTime, h, icon, toast, truncate } from './util.js';

const MAX_ITEMS = 50;

function sanitize(raw) {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((it) => it && typeof it.text === 'string' && it.text.trim())
    .map((it) => ({ text: it.text, at: Number(it.at) || Date.now() }))
    .slice(0, MAX_ITEMS);
}

export function createHistory({ speech }) {
  const els = collect({ list: 'history-list', empty: 'history-empty', clear: 'history-clear', count: 'history-count' });
  let items = sanitize(load(KEYS.history, []));
  if (!els) return { add() {} };

  function persist() {
    save(KEYS.history, items);
  }

  function render() {
    els.empty.hidden = items.length > 0;
    els.clear.hidden = items.length === 0;
    els.count.textContent = items.length ? `(${items.length})` : '';
    els.list.replaceChildren(...items.map((item, index) => {
      const short = truncate(item.text, 40);
      return h('li', { class: 'history-item' },
        h('p', { class: 'history-item__text', text: item.text }),
        h('time', { class: 'history-item__time', datetime: new Date(item.at).toISOString(), text: formatTime(item.at) }),
        h('div', { class: 'history-item__actions' },
          h('button', { type: 'button', class: 'icon-btn icon-btn--sm', title: 'Speak', 'aria-label': `Speak “${short}”`, dataset: { action: 'speak', index }, html: icon('speaker') }),
          h('button', { type: 'button', class: 'icon-btn icon-btn--sm', title: 'Copy', 'aria-label': `Copy “${short}”`, dataset: { action: 'copy', index }, html: icon('copy') }),
          h('button', { type: 'button', class: 'icon-btn icon-btn--sm', title: 'Delete', 'aria-label': `Delete “${short}” from history`, dataset: { action: 'delete', index }, html: icon('trash') })));
    }));
  }

  els.list.addEventListener('click', async (event) => {
    const btn = event.target instanceof Element ? event.target.closest('button[data-action]') : null;
    if (!btn) return;
    const item = items[Number(btn.dataset.index)];
    if (!item) return;
    switch (btn.dataset.action) {
      case 'speak':
        if (!speech.supported) toast("Speech isn't available in this browser.", { type: 'error' });
        else speech.speak(item.text.toLowerCase(), { interrupt: true });
        break;
      case 'copy': {
        const ok = await copyText(item.text);
        toast(ok ? 'Copied to the clipboard.' : "Couldn't copy. Select the text and copy it yourself.", { type: ok ? 'success' : 'error' });
        break;
      }
      case 'delete': {
        items = items.filter((it) => it !== item);
        persist();
        render();
        // keep keyboard focus in the list
        const next = els.list.querySelector('button[data-action="delete"]');
        (next || els.empty).focus?.();
        break;
      }
      default:
        break;
    }
  });

  els.clear.addEventListener('click', async () => {
    if (!items.length) return;
    const ok = await confirmDialog({
      title: 'Clear history?',
      body: `This removes all ${items.length} saved ${items.length === 1 ? 'sentence' : 'sentences'} from this browser.`,
      confirmText: 'Clear history',
      danger: true,
    });
    if (!ok) return;
    items = [];
    persist();
    render();
    toast('History cleared.');
  });

  render();

  return {
    /** Add text unless it is empty or the same as the newest entry. */
    add(text) {
      const t = String(text ?? '').trim();
      if (!t || (items[0] && items[0].text === t)) return;
      items.unshift({ text: t, at: Date.now() });
      items = items.slice(0, MAX_ITEMS);
      persist();
      render();
    },
  };
}
