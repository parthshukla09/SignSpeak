// Sends camera frames to the server: the UNMIRRORED video, scaled to at most 640 px wide, as
// JPEG over the WebSocket. Exactly one frame is in flight; the next is sent after its 'result'.

const MIN_INTERVAL_MS = 30;
const MAX_WIDTH = 640;
const JPEG_QUALITY = 0.75;

export class FrameSender {
  /**
   * @param {object} o
   * @param {HTMLVideoElement} o.video
   * @param {import('./connection.js').Connection} o.conn
   * @param {() => boolean} o.isActive  whether the current view wants recognition
   */
  constructor({ video, conn, isActive }) {
    this.video = video;
    this.conn = conn;
    this.isActive = isActive;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.encoding = false;
    this.timer = 0;
    this.lastSent = 0;
    // Safety net: picks sending back up after ready / camera start / tab becoming visible.
    setInterval(() => this.pump(), 250);
    document.addEventListener('visibilitychange', () => this.pump());
  }

  /** Send a frame if everything allows it (cheap to call often). */
  pump() {
    if (this.timer || this.encoding || !this.ctx) return;
    if (document.hidden || !this.isActive() || !this.conn.canSendFrame()) return;
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth || !v.videoHeight) return;
    const wait = this.lastSent + MIN_INTERVAL_MS - performance.now();
    if (wait > 0) {
      this.timer = setTimeout(() => {
        this.timer = 0;
        this.pump();
      }, wait);
      return;
    }
    this._capture(v);
  }

  _capture(v) {
    const scale = Math.min(1, MAX_WIDTH / v.videoWidth);
    const w = Math.max(1, Math.round(v.videoWidth * scale));
    const h = Math.max(1, Math.round(v.videoHeight * scale));
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;
    try {
      this.ctx.drawImage(v, 0, 0, w, h); // the raw video element: never mirrored
    } catch {
      return;
    }
    this.encoding = true;
    this.lastSent = performance.now();
    const done = (blob) => {
      this.encoding = false;
      if (!blob || !blob.size || !this.conn.sendFrame(blob)) {
        // Not sent (socket closed or loading): the interval pump retries.
        return;
      }
    };
    try {
      this.canvas.toBlob(done, 'image/jpeg', JPEG_QUALITY);
    } catch {
      this.encoding = false;
    }
  }
}
