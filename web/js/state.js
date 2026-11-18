// Shared application state and a tiny event bus.

import { LETTERS } from './util.js';

export class Emitter {
  constructor() {
    this.handlers = new Map();
  }

  on(event, fn) {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set());
    this.handlers.get(event).add(fn);
    return () => this.handlers.get(event)?.delete(fn);
  }

  emit(event, data) {
    const set = this.handlers.get(event);
    if (!set) return;
    for (const fn of [...set]) {
      try {
        fn(data);
      } catch (err) {
        console.error(`[SignSpeak] handler for "${event}" failed`, err);
      }
    }
  }
}

/** The event bus. Server messages arrive as `msg:<type>` (msg:ready, msg:result, ...). */
export const bus = new Emitter();

/** Plain shared state. Modules read it freely; the owner of each field writes it. */
export const store = {
  server: null,          // last GET /api/state payload (null while unknown)
  recordings: null,      // {letter: {takes, images}} (null while unknown)
  modelId: null,         // model in use (from 'ready')
  modelLabel: '',
  loadingModel: null,    // model id being loaded (from 'loading'), null otherwise
  config: null,          // full recognition config from the server
  classes: LETTERS.slice(),
  text: '',              // composed text (server is the source of truth)
  result: null,          // last frame result
  camera: 'starting',    // 'starting' | 'live' | 'error'
};
