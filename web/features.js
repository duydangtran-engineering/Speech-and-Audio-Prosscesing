// Log-mel front end for the browser.
//
// This must reproduce src/features.py exactly. A model that scores 91%
// offline drops to guessing if the browser computes even slightly
// different features, and the failure is silent - the page runs, the
// predictions are just wrong. web/parity.html checks the two against a
// reference tensor exported from Python.
//
// Matches torchaudio.transforms.MelSpectrogram with:
//   sample_rate 16000, n_fft 512, win_length 400, hop_length 160,
//   n_mels 40, f_min 20, f_max 7600, power 2.0, center true,
//   pad_mode "reflect", norm null, mel_scale "htk"

export const FEATURE_CONFIG = {
  sampleRate: 16000,
  clipSamples: 16000,
  nFft: 512,
  winLength: 400,
  hopLength: 160,
  nMels: 40,
  fMin: 20.0,
  fMax: 7600.0,
  nFrames: 101,
  logEpsilon: 1e-6,
};

// --- FFT ---------------------------------------------------------------
// Iterative radix-2 Cooley-Tukey. n must be a power of two, which is the
// whole reason n_fft is 512 and not the 400-sample analysis window.
class FFT {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error(`FFT size ${n} is not a power of two`);
    this.n = n;
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
  }

  // In-place transform of (re, im); both Float64Array(n).
  transform(re, im) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let i = 0; i < n; i += size) {
        for (let j = i, k = 0; j < i + half; j++, k += step) {
          const c = cos[k], s = sin[k];
          const lRe = re[j + half], lIm = im[j + half];
          const tRe = lRe * c - lIm * s;
          const tIm = lRe * s + lIm * c;
          re[j + half] = re[j] - tRe;
          im[j + half] = im[j] - tIm;
          re[j] += tRe;
          im[j] += tIm;
        }
      }
    }
  }
}

// --- Mel filterbank ----------------------------------------------------
const hzToMel = (hz) => 2595.0 * Math.log10(1.0 + hz / 700.0);
const melToHz = (mel) => 700.0 * (10.0 ** (mel / 2595.0) - 1.0);

// Reproduces torchaudio.functional.melscale_fbanks(norm=None, mel_scale="htk").
function melFilterbank(cfg) {
  const nFreqs = cfg.nFft / 2 + 1;
  const allFreqs = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) {
    allFreqs[i] = (i * cfg.sampleRate) / cfg.nFft;
  }

  const mMin = hzToMel(cfg.fMin);
  const mMax = hzToMel(cfg.fMax);
  const fPts = new Float64Array(cfg.nMels + 2);
  for (let i = 0; i < cfg.nMels + 2; i++) {
    fPts[i] = melToHz(mMin + ((mMax - mMin) * i) / (cfg.nMels + 1));
  }

  // filters[m] holds one triangular filter over the frequency bins.
  const filters = [];
  for (let m = 0; m < cfg.nMels; m++) {
    const row = new Float64Array(nFreqs);
    const left = fPts[m], center = fPts[m + 1], right = fPts[m + 2];
    for (let k = 0; k < nFreqs; k++) {
      const down = (allFreqs[k] - left) / (center - left);
      const up = (right - allFreqs[k]) / (right - center);
      row[k] = Math.max(0, Math.min(down, up));
    }
    filters.push(row);
  }
  return filters;
}

// Periodic Hann of win_length, zero-padded to n_fft and centred - this is
// what torch.stft does internally when win_length < n_fft.
function paddedHann(cfg) {
  const window = new Float64Array(cfg.nFft);
  const offset = Math.floor((cfg.nFft - cfg.winLength) / 2);
  for (let i = 0; i < cfg.winLength; i++) {
    window[offset + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / cfg.winLength);
  }
  return window;
}

// --- Extractor ---------------------------------------------------------
export class LogMelExtractor {
  constructor(config = FEATURE_CONFIG) {
    this.cfg = { ...FEATURE_CONFIG, ...config };
    this.fft = new FFT(this.cfg.nFft);
    this.filters = melFilterbank(this.cfg);
    this.window = paddedHann(this.cfg);
    this.re = new Float64Array(this.cfg.nFft);
    this.im = new Float64Array(this.cfg.nFft);
  }

  // Reflect padding, matching torch.stft(center=True, pad_mode="reflect").
  // Index -1 maps to sample 1, not sample 0.
  static reflectPad(signal, pad) {
    const out = new Float64Array(signal.length + 2 * pad);
    for (let i = 0; i < out.length; i++) {
      let j = i - pad;
      const n = signal.length;
      // Fold repeatedly, so a pad wider than the signal still works.
      while (j < 0 || j >= n) {
        if (j < 0) j = -j;
        if (j >= n) j = 2 * (n - 1) - j;
      }
      out[i] = signal[j];
    }
    return out;
  }

  /**
   * waveform: Float32Array | Float64Array of clipSamples
   * returns: Float32Array of nMels * nFrames, row-major (mel, frame)
   */
  extract(waveform) {
    const cfg = this.cfg;
    const pad = Math.floor(cfg.nFft / 2);
    const padded = LogMelExtractor.reflectPad(waveform, pad);
    const nFrames = Math.floor((padded.length - cfg.nFft) / cfg.hopLength) + 1;
    const out = new Float32Array(cfg.nMels * nFrames);

    const power = new Float64Array(cfg.nFft / 2 + 1);

    for (let t = 0; t < nFrames; t++) {
      const start = t * cfg.hopLength;
      for (let i = 0; i < cfg.nFft; i++) {
        this.re[i] = padded[start + i] * this.window[i];
        this.im[i] = 0;
      }
      this.fft.transform(this.re, this.im);
      for (let k = 0; k < power.length; k++) {
        power[k] = this.re[k] * this.re[k] + this.im[k] * this.im[k];
      }
      for (let m = 0; m < cfg.nMels; m++) {
        const row = this.filters[m];
        let sum = 0;
        for (let k = 0; k < power.length; k++) sum += row[k] * power[k];
        out[m * nFrames + t] = Math.log(sum + cfg.logEpsilon);
      }
    }

    // Per-utterance normalization: same statistics the model saw during
    // training, so overall recording level stops mattering.
    let mean = 0;
    for (let i = 0; i < out.length; i++) mean += out[i];
    mean /= out.length;
    let variance = 0;
    for (let i = 0; i < out.length; i++) {
      const d = out[i] - mean;
      variance += d * d;
    }
    // Unbiased variance, matching torch.std default (correction = 1).
    const std = Math.max(Math.sqrt(variance / (out.length - 1)), 1e-5);
    for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) / std;

    return out;
  }
}
