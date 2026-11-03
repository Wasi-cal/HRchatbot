// AudioWorklet sources (the worklet scope has no imports, so they are strings loaded through a Blob URL).
export const AGENT_RATE = 24000;

/** Mic capture: resamples to the agent's input rate (linear interpolation) and posts ~50 ms linear16 chunks. */
export const CAPTURE_SRC = `
class Pcm extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    this.target = opts.processorOptions.targetRate;
    this.ratio = sampleRate / this.target;
    this.pos = 0;
    this.out = [];
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    while (this.pos < ch.length) {
      const i = Math.floor(this.pos), f = this.pos - i;
      const a = ch[i], b = i + 1 < ch.length ? ch[i + 1] : a;
      this.out.push(a + (b - a) * f);
      this.pos += this.ratio;
    }
    this.pos -= ch.length;
    const frame = Math.round(this.target * 0.05);
    while (this.out.length >= frame) {
      const chunk = this.out.splice(0, frame);
      const pcm = new Int16Array(frame);
      for (let k = 0; k < frame; k++) {
        const s = Math.max(-1, Math.min(1, chunk[k]));
        pcm[k] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-capture", Pcm);
`;

/** Playback: a thin processor around AudioCore.PlaybackCore (public/audio_core.js is prepended to this source). */
export const PLAYER_SRC = `
class Player extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const p = opts.processorOptions;
    this.core = new globalThis.AudioCore.PlaybackCore({ inRate: p.inRate, outRate: sampleRate, prebufferMs: p.prebufferMs, fadeMs: 8 });
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.pcm) this.core.write(new Int16Array(m.pcm));
      else if (m.flush) this.core.flush();
      else if (m.end) this.core.endOfStream();
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    this.core.read(out);
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    return true;
  }
}
registerProcessor("pcm-player", Player);
`;
