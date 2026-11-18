// WebSocket client for /ws (see web/PROTOCOL.md).
//
// Status: 'connecting' -> (open, hello sent) -> 'loading' -> 'ready'; 'offline' while waiting to
// reconnect. Every server message is re-emitted on the bus as `msg:<type>`.

// Frames: the server answers every binary message with exactly one 'result' (a 'skipped' one
// when it can't process it) and handles messages in order, so a count of unanswered frames is
// exact. A frame sent just before a model switch waits on the server behind the whole load, and
// the first frame after a load can take seconds; the count stays right through both, where a
// "sent / answered" flag reset on 'ready' or on a timeout would leave two frames in flight.
const RETRY_DELAYS_MS = [1000, 2000, 3000, 5000, 8000, 10000];
const FRAME_TIMEOUT_MS = 5000;     // no result this long: allow one extra frame (never more)
const MAX_BUFFERED_BYTES = 1_500_000;

export class Connection {
  constructor({ bus, getHello }) {
    this.bus = bus;
    this.getHello = getHello;
    this.ws = null;
    this.status = 'connecting';
    this.ready = false;        // a model is loaded and no other load is in progress
    this.hasEngine = false;    // a 'ready' arrived on this socket (commands are accepted)
    this.outstanding = 0;      // frames sent on this socket whose 'result' hasn't arrived
    this.sentAt = 0;
    this.attempt = 0;
    this.retryTimer = 0;
    this.retryAt = 0;
    this.connId = 0;
    this.everReady = false;    // any 'ready' on any socket (to tell reconnects apart)
    this.helloModel = null;
  }

  get url() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/ws`;
  }

  get isOpen() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  /** Open and usable for commands (mode, command, learn_*, record_*). */
  get live() {
    return this.isOpen && this.hasEngine;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) return;
    clearTimeout(this.retryTimer);
    this.retryTimer = 0;
    this._setStatus('connecting');

    let ws;
    try {
      ws = new WebSocket(this.url);
    } catch (err) {
      console.warn('[SignSpeak] WebSocket failed to open', err);
      this._scheduleRetry();
      return;
    }
    const id = ++this.connId;
    this.ws = ws;
    this.ready = false;
    this.hasEngine = false;
    this.outstanding = 0;
    ws.binaryType = 'arraybuffer';

    ws.addEventListener('open', () => {
      if (id !== this.connId) return;
      this.attempt = 0;
      let hello;
      try {
        hello = this.getHello();
      } catch (err) {
        console.error(err);
        hello = { type: 'hello', model: null, config: {} };
      }
      this.helloModel = hello.model ?? null;
      this.send(hello);
      this.bus.emit('ws:open', { connId: id });
    });

    ws.addEventListener('message', (event) => {
      if (id !== this.connId || typeof event.data !== 'string') return;
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        console.warn('[SignSpeak] ignoring a message that is not JSON');
        return;
      }
      if (msg && typeof msg === 'object' && typeof msg.type === 'string') this._handle(msg);
    });

    ws.addEventListener('close', () => {
      if (id !== this.connId) return;
      const wasReady = this.hasEngine;
      this.ws = null;
      this.ready = false;
      this.hasEngine = false;
      this.outstanding = 0;
      this.bus.emit('ws:close', { connId: id, wasReady });
      this._scheduleRetry();
    });

    ws.addEventListener('error', () => { /* a 'close' event always follows */ });
  }

  _handle(msg) {
    let payload = msg;
    switch (msg.type) {
      case 'loading':
        this.ready = false;
        this._setStatus('loading');
        break;
      case 'ready': {
        const fresh = !this.hasEngine;          // first ready on this socket
        const reconnect = fresh && this.everReady;
        this.ready = true;
        this.hasEngine = true;
        this.everReady = true;
        // Not resetting `outstanding`: a frame queued behind the load is still owed a result.
        payload = { ...msg, fresh, reconnect, requestedModel: fresh ? this.helloModel : undefined };
        this._setStatus('ready');
        break;
      }
      case 'result':
        this.outstanding = Math.max(0, this.outstanding - 1);
        break;
      case 'error':
        // e.g. a model failed to load: the server keeps the previous one, so frames can resume
        if (this.hasEngine && !this.ready && this.status === 'loading') {
          this.ready = true;
          this._setStatus('ready');
        }
        break;
      default:
        break;
    }
    this.bus.emit(`msg:${msg.type}`, payload);
  }

  _setStatus(status) {
    if (this.status === status) return;
    this.status = status;
    this.bus.emit('ws:status', { status, retryAt: this.retryAt });
  }

  _scheduleRetry() {
    const delay = RETRY_DELAYS_MS[Math.min(this.attempt, RETRY_DELAYS_MS.length - 1)];
    this.attempt += 1;
    this.retryAt = Date.now() + delay;
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => this.connect(), delay);
    this.status = 'offline';
    this.bus.emit('ws:status', { status: 'offline', retryAt: this.retryAt });
  }

  /** Reconnect immediately (the "Retry now" button). */
  retryNow() {
    if (this.ws) return; // already connecting or open
    this.attempt = Math.max(0, this.attempt - 1);
    this.connect();
  }

  /** Send a JSON message. Returns false if the socket is not open. */
  send(obj) {
    if (!this.isOpen) return false;
    try {
      this.ws.send(JSON.stringify(obj));
      return true;
    } catch (err) {
      console.warn('[SignSpeak] send failed', err);
      return false;
    }
  }

  /**
   * Switch model. Frames stop at once (not only when 'loading' arrives): one sent in between
   * would sit on the server behind the load.
   */
  switchModel(id) {
    if (!this.send({ type: 'model', model: id })) return false;
    this.ready = false;
    this._setStatus('loading');
    return true;
  }

  /** Whether a camera frame may be sent now: ready, and no frame in flight. */
  canSendFrame() {
    if (!this.isOpen || !this.ready) return false;
    if (this.outstanding > 0) {
      // Normally wait for the result. If none has come for a long time, allow a single extra
      // frame; it is counted, so sending returns to one in flight once the results arrive.
      if (this.outstanding > 1 || performance.now() - this.sentAt < FRAME_TIMEOUT_MS) return false;
    }
    return this.ws.bufferedAmount < MAX_BUFFERED_BYTES;
  }

  sendFrame(blob) {
    if (!this.canSendFrame()) return false;
    try {
      this.ws.send(blob);
      this.outstanding += 1;
      this.sentAt = performance.now();
      return true;
    } catch (err) {
      console.warn('[SignSpeak] frame send failed', err);
      return false;
    }
  }
}
