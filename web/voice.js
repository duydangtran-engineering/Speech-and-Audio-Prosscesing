// Microphone capture and keyword spotting in the browser.
//
// Mirrors src/live_demo.py exactly, including the parts that were added
// after watching it misbehave: a VAD gate, a confidence threshold, a
// refractory period, and the rule that the detector re-arms only after a
// silent window. Without that last one a long word fires twice, because
// it is still inside the one-second buffer when the timer expires.

import { LogMelExtractor } from './features.js';

export class VoiceController {
  constructor(meta, session) {
    this.meta = meta;
    this.session = session;
    this.labels = meta.labels;
    this.commandIds = new Set(
      this.labels.map((l, i) => (l.startsWith('_') ? -1 : i)).filter(i => i >= 0));

    const r = meta.runtime;
    this.threshold = r.threshold;
    this.smoothing = r.smoothing;
    this.hop = r.hop;
    this.vadRms = r.vadRms;
    this.refractory = r.refractory;

    this.extractor = new LogMelExtractor(meta.features);
    this.ring = new Float32Array(meta.features.clipSamples);
    this.history = [];
    this.armed = true;
    this.lastFired = 0;
    this.lastLabel = null;
    this.latencies = [];

    this.onCommand = () => {};
    this.onFrame = () => {};
  }

  async start(deviceId = null) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        echoCancellation: false,   // these process speech in ways the
        noiseSuppression: false,   // model never saw during training
        autoGainControl: false,
      },
    });

    // Asking for 16 kHz directly avoids resampling; browsers that refuse
    // fall back to their own rate and we resample per block.
    this.audio = new AudioContext({ sampleRate: this.meta.features.sampleRate });
    this.sourceRate = this.audio.sampleRate;
    this.resampleRatio = this.sourceRate / this.meta.features.sampleRate;

    const blockSamples = Math.round(this.hop * this.sourceRate);
    const workletCode = `
      class Capture extends AudioWorkletProcessor {
        constructor() { super(); this.buffer = []; this.target = ${blockSamples}; }
        process(inputs) {
          const input = inputs[0][0];
          if (!input) return true;
          this.buffer.push(...input);
          while (this.buffer.length >= this.target) {
            this.port.postMessage(
              Float32Array.from(this.buffer.splice(0, this.target)));
          }
          return true;
        }
      }
      registerProcessor('capture', Capture);
    `;
    const url = URL.createObjectURL(
      new Blob([workletCode], { type: 'application/javascript' }));
    await this.audio.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const source = this.audio.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(this.audio, 'capture');
    node.port.onmessage = (event) => this.push(event.data);
    source.connect(node);
    this.running = true;
    return this.sourceRate;
  }

  static resample(block, ratio) {
    if (Math.abs(ratio - 1) < 1e-6) return block;
    const n = Math.round(block.length / ratio);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const pos = i * ratio;
      const j = Math.floor(pos);
      const frac = pos - j;
      out[i] = (block[j] ?? 0) * (1 - frac) + (block[j + 1] ?? block[j] ?? 0) * frac;
    }
    return out;
  }

  async push(rawBlock) {
    if (!this.running || this.busy) return;
    const block = VoiceController.resample(rawBlock, this.resampleRatio);
    const n = Math.min(block.length, this.ring.length);
    this.ring.copyWithin(0, n);
    this.ring.set(block.subarray(0, n), this.ring.length - n);

    let sum = 0;
    for (let i = 0; i < this.ring.length; i++) sum += this.ring[i] * this.ring[i];
    const rms = Math.sqrt(sum / this.ring.length);

    if (rms < this.vadRms) {
      this.history.length = 0;
      this.armed = true;                 // a silent window ends the utterance
      this.lastLabel = null;
      this.onFrame({ rms, label: null, confidence: 0, silent: true });
      return;
    }

    this.busy = true;
    try {
      const began = performance.now();
      const mel = this.extractor.extract(this.ring);
      const feeds = {
        logmel: new ort.Tensor('float32', mel,
          [1, 1, this.meta.features.nMels, this.meta.features.nFrames]),
      };
      const output = await this.session.run(feeds);
      const logits = output.logits.data;
      this.latencies.push(performance.now() - began);

      let max = -Infinity;
      for (const v of logits) if (v > max) max = v;
      let total = 0;
      const probs = new Float64Array(logits.length);
      for (let i = 0; i < logits.length; i++) {
        probs[i] = Math.exp(logits[i] - max);
        total += probs[i];
      }
      for (let i = 0; i < probs.length; i++) probs[i] /= total;

      this.history.push(probs);
      if (this.history.length > Math.max(1, this.smoothing)) this.history.shift();
      const smoothed = new Float64Array(probs.length);
      for (const h of this.history) {
        for (let i = 0; i < h.length; i++) smoothed[i] += h[i] / this.history.length;
      }

      let best = 0;
      for (let i = 1; i < smoothed.length; i++) {
        if (smoothed[i] > smoothed[best]) best = i;
      }
      const confidence = smoothed[best];
      const label = this.labels[best];
      this.onFrame({ rms, label, confidence, silent: false });

      // "One utterance, one command" is enforced by requiring a silent
      // window before the SAME word can fire again. A *different* word
      // cannot be an echo of the one just fired, so it is allowed
      // through immediately - that is what makes "trai, phai" spoken
      // back to back both land, which the fixed lockout used to swallow.
      const now = performance.now() / 1000;
      const isNewWord = label !== this.lastLabel;
      if (this.commandIds.has(best) && confidence >= this.threshold
          && (this.armed || isNewWord)
          && now - this.lastFired >= this.refractory) {
        this.lastFired = now;
        this.lastLabel = label;
        this.armed = false;
        this.history.length = 0;
        this.ring.fill(0);
        this.onCommand(label, confidence);
      }
    } finally {
      this.busy = false;
    }
  }

  stats() {
    if (!this.latencies.length) return null;
    const sorted = [...this.latencies].sort((a, b) => a - b);
    const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
    return {
      count: sorted.length,
      mean,
      p95: sorted[Math.floor(sorted.length * 0.95)],
      endToEnd: this.hop * 1000 * this.smoothing + mean,
    };
  }
}
