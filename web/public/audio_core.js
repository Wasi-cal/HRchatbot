// Pure audio-playback logic, no Web Audio / DOM dependency (unit-tested in Node: eval/ui/audio_core_test.js).
// Loaded as a plain <script> in the page (exposes window.AudioCore) and its source text is prepended to the AudioWorklet
// code (the worklet scope has no imports), so keep it self-contained.
(function (root) {
  "use strict";

  // PCM16 little-endian needs 2 bytes per sample. WebSocket frames can have any length, so a trailing odd byte is
  // carried into the next frame; dropping it would shift every following sample by one byte (loud noise).
  class Pcm16Aligner {
    constructor() { this.carry = -1; this.oddChunks = 0; }
    // bytes: Uint8Array -> Int16Array of whole samples (possibly empty)
    push(bytes) {
      let n = bytes.length;
      let offset = 0;
      let out;
      const hasCarry = this.carry >= 0;
      const total = n + (hasCarry ? 1 : 0);
      const whole = total >> 1;
      const buf = new Uint8Array(whole * 2);
      let w = 0;
      if (hasCarry) { buf[w++] = this.carry; this.carry = -1; }
      const take = whole * 2 - w;
      for (let i = 0; i < take; i++) buf[w++] = bytes[offset++];
      if (offset < n) { this.carry = bytes[offset]; this.oddChunks++; }
      out = new Int16Array(buf.buffer, 0, whole);
      return out;
    }
    reset() { this.carry = -1; }
  }

  // Jitter buffer + one-time resampler + click-free start/stop. Pull model: the audio thread calls read(out) every render
  // quantum. Input is PCM16 at inRate; output is float32 at outRate (equal rates -> straight copy, no interpolation).
  //  - prebuffer: playback (re)starts only once `prebufferMs` of audio is queued (or end-of-stream / maxWaitMs reached)
  //  - underrun: queue ran dry mid-utterance -> short fade to 0, silence, re-prebuffer; counted
  //  - flush(): barge-in; fades the current output to 0 over fadeMs, then drops everything still queued
  class PlaybackCore {
    constructor(opts) {
      opts = opts || {};
      this.inRate = opts.inRate || 24000;
      this.outRate = opts.outRate || this.inRate;
      this.step = this.inRate / this.outRate;
      this.capacity = Math.round((opts.bufferSeconds || 60) * this.inRate);
      this.ring = new Float32Array(this.capacity);
      this.head = 0;            // read index (integer part)
      this.frac = 0;            // read position fraction between ring[head] and ring[head+1]
      this.count = 0;           // samples queued
      this.prebuffer = Math.round(((opts.prebufferMs == null ? 100 : opts.prebufferMs) / 1000) * this.inRate);
      this.maxWait = Math.round(((opts.maxWaitMs == null ? 300 : opts.maxWaitMs) / 1000) * this.outRate);
      this.fadeSamples = Math.max(1, Math.round(((opts.fadeMs == null ? 8 : opts.fadeMs) / 1000) * this.outRate));
      this.rampSamples = Math.max(1, Math.round(0.004 * this.outRate));  // fade-in after (re)start / fade-out on underrun
      this.playing = false;
      this.waited = 0;
      this.ended = false;       // end-of-stream seen: draining is not an underrun
      this.fadeLeft = 0;        // >0 while a flush fade-out is running
      this.rampIn = 0;          // samples of fade-in left
      this.last = 0;            // last output sample (for continuity on underrun)
      this.stats = { underruns: 0, overflow: 0, flushes: 0, samplesIn: 0, samplesOut: 0, maxQueued: 0 };
    }

    queuedMs() { return (this.count / this.inRate) * 1000; }

    write(int16) {
      const n = int16.length;
      if (!n) return;
      this.ended = false;
      let room = this.capacity - this.count;
      let m = n;
      if (m > room) { this.stats.overflow += m - room; m = room; }
      let w = (this.head + this.count) % this.capacity;
      for (let i = 0; i < m; i++) {
        this.ring[w] = int16[i] / 32768;
        if (++w === this.capacity) w = 0;
      }
      this.count += m;
      this.stats.samplesIn += m;
      if (this.count > this.stats.maxQueued) this.stats.maxQueued = this.count;
    }

    endOfStream() { this.ended = true; }

    flush() {
      this.stats.flushes++;
      this.ended = false;
      if (this.playing) this.fadeLeft = this.fadeSamples;   // fade whatever is playing, then drop the queue
      else this._drop();
    }

    _drop() { this.head = 0; this.frac = 0; this.count = 0; this.playing = false; this.waited = 0; this.fadeLeft = 0; this.rampIn = 0; }

    // Fills out (Float32Array) completely; returns out.
    read(out) {
      const len = out.length;
      let i = 0;
      while (i < len) {
        if (!this.playing) {
          const ready = this.count >= this.prebuffer || (this.ended && this.count > 0) || (this.count > 0 && this.waited >= this.maxWait);
          if (!ready) {
            if (this.count > 0) this.waited += len - i;
            while (i < len) out[i++] = 0;
            break;
          }
          this.playing = true; this.waited = 0; this.rampIn = this.rampSamples; this.last = 0;
        }
        // need one sample ahead for interpolation unless rates are equal
        const needed = this.step === 1 ? 1 : 2;
        if (this.count < needed + 0 && !(this.step !== 1 && this.count === 1 && this.ended)) {
          // ran dry
          if (!this.ended && this.fadeLeft === 0) this.stats.underruns++;
          this._tail(out, i);
          i = len;
          this._dryStop();
          break;
        }
        let s;
        if (this.step === 1) {
          s = this.ring[this.head];
        } else {
          const a = this.ring[this.head];
          const b = this.ring[(this.head + 1) % this.capacity];
          s = a + (b - a) * this.frac;
        }
        if (this.rampIn > 0) { s *= 1 - this.rampIn / (this.rampSamples + 1); this.rampIn--; }
        if (this.fadeLeft > 0) {
          s *= this.fadeLeft / (this.fadeSamples + 1);
          if (--this.fadeLeft === 0) { out[i++] = s; this.last = s; this.stats.samplesOut++; this._drop(); this._silence(out, i); i = len; break; }
        }
        out[i++] = s;
        this.last = s;
        this.stats.samplesOut++;
        this.frac += this.step;
        const adv = Math.floor(this.frac);
        if (adv > 0) {
          this.frac -= adv;
          const take = Math.min(adv, this.count);
          this.head = (this.head + take) % this.capacity;
          this.count -= take;
        }
      }
      return out;
    }

    // underrun: decay the last value to 0 instead of stepping (a step is an audible tick)
    _tail(out, from) {
      const n = Math.min(this.rampSamples, out.length - from);
      for (let k = 0; k < n; k++) out[from + k] = this.last * (1 - (k + 1) / (n + 1));
      for (let k = from + n; k < out.length; k++) out[k] = 0;
    }
    _silence(out, from) { for (let k = from; k < out.length; k++) out[k] = 0; }
    _dryStop() { this.playing = false; this.waited = 0; this.frac = 0; this.last = 0; }
  }

  class ChunkStats {
    constructor() { this.chunks = 0; this.odd = 0; this.bytes = 0; this.hist = { "<=480": 0, "<=1920": 0, "<=4800": 0, ">4800": 0 }; this.lastAt = 0; this.maxGapMs = 0; this.gapsOver = 0; }
    add(byteLength, nowMs, gapThresholdMs) {
      this.chunks++; this.bytes += byteLength;
      if (byteLength & 1) this.odd++;
      this.hist[byteLength <= 480 ? "<=480" : byteLength <= 1920 ? "<=1920" : byteLength <= 4800 ? "<=4800" : ">4800"]++;
      if (this.lastAt) {
        const gap = nowMs - this.lastAt;
        if (gap > this.maxGapMs) this.maxGapMs = gap;
        if (gap > gapThresholdMs) this.gapsOver++;
      }
      this.lastAt = nowMs;
    }
    resetBurst() { this.lastAt = 0; }  // call between utterances so silence between answers is not a "gap"
  }

  function wavBytes(int16Samples, sampleRate) {
    const dataLen = int16Samples.length * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + dataLen, true); str(8, "WAVE"); str(12, "fmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, "data"); v.setUint32(40, dataLen, true);
    new Int16Array(buf, 44, int16Samples.length).set(int16Samples);
    return new Uint8Array(buf);
  }

  const api = { Pcm16Aligner, PlaybackCore, ChunkStats, wavBytes };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AudioCore = api;
})(typeof globalThis !== "undefined" ? globalThis : typeof self !== "undefined" ? self : this);
