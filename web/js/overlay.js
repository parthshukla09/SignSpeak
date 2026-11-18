// Draws hand landmarks and the crop box on a <canvas> laid exactly over the <video>.
//
// Result coordinates are normalised to the frame that was sent and are NOT mirrored. The video
// uses object-fit: contain, so the picture may be letterboxed inside the stage; the same fit is
// computed here. When the view is mirrored, x is drawn at 1 - x.

// MediaPipe hand connections (21 landmarks).
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const HAND_COLORS = ['#38bdf8', '#f472b6'];
const BOX_COLOR = '#22c55e';
const BOX_UNCERTAIN = '#fbbf24';
const REC_COLOR = '#ef4444';
const STALE_MS = 1500;
const FONT = '700 14px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

export class Overlay {
  /**
   * @param {object} o
   * @param {HTMLElement} o.stage   the element the canvas covers
   * @param {HTMLVideoElement} o.video
   * @param {HTMLCanvasElement} o.canvas
   * @param {() => {mirror: boolean, landmarks: boolean}} o.options
   */
  constructor({ stage, video, canvas, options }) {
    this.stage = stage;
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.options = options;
    this.result = null;
    this.resultAt = 0;
    this.raf = 0;
    this.staleTimer = 0;
    this.cssW = 0;
    this.cssH = 0;
    this.dpr = 0;

    const redraw = () => this.request();
    if (typeof ResizeObserver === 'function') new ResizeObserver(redraw).observe(stage);
    window.addEventListener('resize', redraw);
    video.addEventListener('loadedmetadata', redraw);
    video.addEventListener('resize', redraw);
  }

  setResult(result) {
    this.result = result;
    this.resultAt = performance.now();
    this.request();
    clearTimeout(this.staleTimer);
    this.staleTimer = setTimeout(() => this.request(), STALE_MS + 50);
  }

  clear() {
    this.result = null;
    this.request();
  }

  /** Schedule one draw on the next animation frame. */
  request() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      try {
        this._draw();
      } catch (err) {
        console.error('[SignSpeak] overlay draw failed', err);
      }
    });
  }

  _resize() {
    const w = this.stage.clientWidth;
    const h = this.stage.clientHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    if (w === this.cssW && h === this.cssH && dpr === this.dpr) return;
    this.cssW = w;
    this.cssH = h;
    this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(w * dpr));
    this.canvas.height = Math.max(1, Math.round(h * dpr));
  }

  _draw() {
    const ctx = this.ctx;
    if (!ctx) return;
    this._resize();
    const { cssW: w, cssH: h, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const r = this.result;
    if (!w || !h || !r || performance.now() - this.resultAt > STALE_MS) return;

    // Where the video picture actually is inside the stage (object-fit: contain).
    const vw = this.video.videoWidth || 4;
    const vh = this.video.videoHeight || 3;
    const scale = Math.min(w / vw, h / vh);
    const dw = vw * scale;
    const dh = vh * scale;
    const ox = (w - dw) / 2;
    const oy = (h - dh) / 2;
    const { mirror, landmarks } = this.options();
    const px = (x) => ox + (mirror ? 1 - x : x) * dw;
    const py = (y) => oy + y * dh;

    if (landmarks && Array.isArray(r.hands)) this._drawHands(ctx, r.hands, px, py);
    if (Array.isArray(r.box) && r.box.length === 4 && r.box.every(finite)) this._drawBox(ctx, r, px, py, w);
  }

  _drawHands(ctx, hands, px, py) {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    hands.forEach((hand, hi) => {
      if (!Array.isArray(hand) || hand.length < 21) return;
      const pts = hand.map((p) => (Array.isArray(p) && finite(p[0]) && finite(p[1]) ? [px(p[0]), py(p[1])] : null));
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.78)';
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        if (!pts[a] || !pts[b]) continue;
        ctx.moveTo(pts[a][0], pts[a][1]);
        ctx.lineTo(pts[b][0], pts[b][1]);
      }
      ctx.stroke();
      const color = HAND_COLORS[hi % HAND_COLORS.length];
      ctx.fillStyle = color;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
      ctx.lineWidth = 1.2;
      pts.forEach((p, i) => {
        if (!p) return;
        ctx.beginPath();
        ctx.arc(p[0], p[1], FINGERTIPS.has(i) ? 4.5 : 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });
    });
  }

  _drawBox(ctx, r, px, py, stageW) {
    const recording = !!(r.record && r.record.active);
    const xa = px(r.box[0]);
    const xb = px(r.box[2]);
    const x = Math.min(xa, xb);
    const y = py(r.box[1]);
    const bw = Math.abs(xb - xa);
    const bh = py(r.box[3]) - y;
    if (bw <= 1 || bh <= 1) return;
    const color = recording ? REC_COLOR : BOX_COLOR;
    const radius = Math.min(14, bw / 4, bh / 4);

    ctx.save();
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(x, y, bw, bh, radius);
    else ctx.rect(x, y, bw, bh);
    ctx.fillStyle = recording ? 'rgba(239, 68, 68, 0.08)' : 'rgba(34, 197, 94, 0.06)';
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.restore();

    // Label: the model's guess, or the recording progress.
    let label = '';
    let bg = color;
    if (recording) {
      const saved = finite(r.record.saved) ? r.record.saved : 0;
      const needed = finite(r.record.needed) ? r.record.needed : 0;
      label = needed ? `REC ${saved}/${needed}` : 'REC';
    } else if (r.best) {
      label = `${r.best} ${Math.round((Number(r.confidence) || 0) * 100)}%`;
      if (!r.letter) bg = BOX_UNCERTAIN;
    }
    if (!label) return;
    ctx.save();
    ctx.font = FONT;
    const tw = ctx.measureText(label).width;
    const lh = 24;
    const lw = tw + 16;
    let lx = Math.min(Math.max(x, 4), stageW - lw - 4);
    let ly = y - lh - 6;
    if (ly < 4) ly = y + 6; // no room above the box: put it inside
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') ctx.roundRect(lx, ly, lw, lh, 7);
    else ctx.rect(lx, ly, lw, lh);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.fillStyle = recording ? '#ffffff' : '#06130b';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, lx + 8, ly + lh / 2 + 0.5);
    ctx.restore();
  }
}
