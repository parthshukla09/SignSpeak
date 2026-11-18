// Webcam access with friendly, specific error messages.

const CONSTRAINTS = {
  audio: false,
  video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
};

export const CAMERA_ERRORS = Object.freeze({
  denied: {
    title: 'Camera access is blocked',
    text: 'SignSpeak needs your camera to see your signs. Click the camera icon in the address bar (or the site settings next to the address), allow the camera, then press Retry. Video is only sent to the SignSpeak server on this computer.',
  },
  notfound: {
    title: 'No camera found',
    text: 'Connect a webcam, or check that the camera is turned on in Windows Settings → Privacy & security → Camera, then press Retry.',
  },
  busy: {
    title: 'The camera is in use',
    text: 'Another app is using your camera, for example Zoom, Teams, the Camera app or webcam.py. Close it, then press Retry.',
  },
  insecure: {
    title: 'Camera not allowed on this address',
    text: 'Browsers only allow the camera on secure pages. Open http://localhost:8000 or http://127.0.0.1:8000 instead.',
  },
  unsupported: {
    title: "This browser can't use a camera",
    text: 'Use a recent version of Chrome, Edge or Firefox.',
  },
  ended: {
    title: 'The camera stopped',
    text: 'The camera was disconnected or turned off. Reconnect it, then press Retry.',
  },
  unknown: {
    title: "Couldn't start the camera",
    text: 'Something went wrong while starting the camera. Press Retry, or reload the page.',
  },
});

function classify(err) {
  const name = (err && err.name) || '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'denied';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return 'notfound';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'busy';
    default:
      return 'unknown';
  }
}

function waitForVideoSize(video, timeoutMs) {
  if (video.videoWidth > 0) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      video.removeEventListener('loadedmetadata', done);
      video.removeEventListener('resize', done);
      resolve();
    };
    const timer = setTimeout(done, timeoutMs);
    video.addEventListener('loadedmetadata', done);
    video.addEventListener('resize', done);
  });
}

export class Camera {
  /**
   * @param {HTMLVideoElement} video
   * @param {(state: 'starting'|'live'|'error', errorKind: string|null, err?: any) => void} onChange
   */
  constructor(video, onChange) {
    this.video = video;
    this.onChange = onChange;
    this.stream = null;
    this.state = 'starting';
    this.errorKind = null;
    this.pending = null;
  }

  get live() {
    return this.state === 'live' && this.video.readyState >= 2 && this.video.videoWidth > 0;
  }

  start() {
    if (!this.pending) this.pending = this._start().finally(() => { this.pending = null; });
    return this.pending;
  }

  async _start() {
    this.stop();
    this._set('starting', null);
    const md = navigator.mediaDevices;
    if (!md || typeof md.getUserMedia !== 'function') {
      this._set('error', window.isSecureContext === false ? 'insecure' : 'unsupported');
      return false;
    }
    let stream = null;
    let error = null;
    try {
      stream = await md.getUserMedia(CONSTRAINTS);
    } catch (err) {
      error = err;
      if (err && (err.name === 'OverconstrainedError' || err.name === 'ConstraintNotSatisfiedError')) {
        try {
          stream = await md.getUserMedia({ video: true, audio: false });
          error = null;
        } catch (err2) {
          error = err2;
        }
      }
    }
    if (!stream) {
      console.warn('[SignSpeak] camera error', error);
      this._set('error', classify(error), error);
      return false;
    }

    this.stream = stream;
    const [track] = stream.getVideoTracks();
    if (track) {
      track.addEventListener('ended', () => {
        if (this.stream !== stream) return;
        this.stop();
        this._set('error', 'ended');
      });
    }
    try {
      this.video.srcObject = stream;
    } catch {
      this._set('error', 'unsupported');
      return false;
    }
    try {
      await this.video.play();
    } catch {
      /* muted autoplay normally succeeds; frames still flow once it does */
    }
    await waitForVideoSize(this.video, 4000);
    if (this.stream !== stream) return false; // stopped meanwhile
    this._set('live', null);
    return true;
  }

  stop() {
    if (this.stream) {
      for (const t of this.stream.getTracks()) {
        try { t.stop(); } catch { /* ignore */ }
      }
      this.stream = null;
    }
    try {
      this.video.srcObject = null;
    } catch {
      /* ignore */
    }
  }

  _set(state, errorKind, err) {
    this.state = state;
    this.errorKind = errorKind;
    try {
      this.onChange(state, errorKind, err);
    } catch (e) {
      console.error(e);
    }
  }
}
