// HTTP API (see web/PROTOCOL.md): GET /api/state, POST /api/recordings/{letter}/delete-last,
// POST /api/finetune, GET /api/finetune.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'ApiError';
    this.status = status; // 0 = network error / timeout
  }
}

function detailOf(data) {
  if (!data || typeof data !== 'object') return '';
  const d = data.detail;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map((x) => (x && x.msg) || String(x)).join('; ');
  return '';
}

async function request(method, url, body, timeoutMs = 8000) {
  const ctrl = typeof AbortController === 'function' ? new AbortController() : null;
  const timer = ctrl ? setTimeout(() => ctrl.abort(), timeoutMs) : 0;
  let res;
  try {
    res = await fetch(url, {
      method,
      cache: 'no-store',
      headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctrl ? ctrl.signal : undefined,
    });
  } catch {
    throw new ApiError(0, "Can't reach the SignSpeak server.");
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) {
    throw new ApiError(res.status, detailOf(data) || `Request failed (${res.status} ${res.statusText || ''})`.trim());
  }
  return data;
}

export const api = {
  getState: () => request('GET', '/api/state'),
  deleteLastTake: (letter) => request('POST', `/api/recordings/${encodeURIComponent(letter)}/delete-last`, {}),
  startFinetune: () => request('POST', '/api/finetune', {}),
  getFinetune: () => request('GET', '/api/finetune'),
};
