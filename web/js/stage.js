// The single live-camera stage (video + overlay + status chips). It is moved into whichever view
// shows the camera, and parked (hidden) when the current view doesn't need it.

import { collect, prefersReducedMotion } from './util.js';
import { CAMERA_ERRORS } from './camera.js';

export function createStage() {
  const els = collect({
    stage: 'stage', video: 'video', canvas: 'overlay', park: 'stage-park',
    placeholder: 'stage-placeholder', badge: 'stage-badge', hint: 'stage-hint',
    countdown: 'stage-countdown', status: 'stage-status', statusText: 'stage-status-text',
    error: 'camera-error', errorTitle: 'camera-error-title', errorText: 'camera-error-text',
    retry: 'camera-retry',
  });
  if (!els) return null;

  let slot = null;

  function place(target) {
    const dest = target || els.park;
    if (els.stage.parentElement !== dest) {
      dest.appendChild(els.stage);
      // Moving a playing <video> is fine, but nudge it in case a browser paused it.
      const p = els.video.srcObject ? els.video.play() : null;
      if (p && typeof p.catch === 'function') p.catch(() => {});
    }
    slot = target || null;
  }

  return {
    els,
    /** Whether a view currently shows the stage (frames are only sent then). */
    get shown() { return !!slot; },
    place,
    setMirror(on) { els.stage.classList.toggle('is-mirrored', !!on); },
    setRecording(on) { els.stage.classList.toggle('is-recording', !!on); },
    setBadge(text, kind = 'rec') {
      els.badge.hidden = !text;
      if (!text) return;
      els.badge.dataset.kind = kind;
      if (els.badge.textContent !== text) els.badge.textContent = text;
    },
    setHint(text, kind = 'warn') {
      els.hint.hidden = !text;
      if (!text) return;
      els.hint.dataset.kind = kind;
      if (els.hint.textContent !== text) els.hint.textContent = text;
    },
    setCountdown(value) {
      if (value == null) {
        els.countdown.hidden = true;
        els.countdown.textContent = '';
        return;
      }
      els.countdown.hidden = false;
      els.countdown.textContent = String(value);
      if (!prefersReducedMotion()) {
        els.countdown.classList.remove('is-tick');
        void els.countdown.offsetWidth; // restart the CSS animation
        els.countdown.classList.add('is-tick');
      }
    },
    setStatus(text) {
      els.status.hidden = !text;
      if (text && els.statusText.textContent !== text) els.statusText.textContent = text;
    },
    setCamera(state, errorKind) {
      els.placeholder.hidden = state !== 'starting';
      const err = state === 'error' ? (CAMERA_ERRORS[errorKind] || CAMERA_ERRORS.unknown) : null;
      els.error.hidden = !err;
      if (err) {
        els.errorTitle.textContent = err.title;
        els.errorText.textContent = err.text;
      }
    },
    /** Clear everything a view may have put on the stage. */
    resetChips() {
      this.setBadge(null);
      this.setHint(null);
      this.setCountdown(null);
      this.setRecording(false);
    },
  };
}
