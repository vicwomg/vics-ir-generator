const TWO_PI = 2 * Math.PI;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function decodeWav(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  assert(view.byteLength >= 12, "The WAV file is truncated.");
  assert(readAscii(view, 0, 4) === "RIFF" && readAscii(view, 8, 4) === "WAVE", "Only RIFF/WAVE files are supported.");

  let format = null;
  let dataOffset = -1;
  let dataLength = 0;
  for (let offset = 12; offset + 8 <= view.byteLength; ) {
    const id = readAscii(view, offset, 4);
    const size = view.getUint32(offset + 4, true);
    const body = offset + 8;
    assert(body + size <= view.byteLength, `The WAV ${id} chunk is truncated.`);
    if (id === "fmt ") {
      assert(size >= 16, "The WAV format chunk is invalid.");
      format = {
        tag: view.getUint16(body, true),
        channels: view.getUint16(body + 2, true),
        sampleRate: view.getUint32(body + 4, true),
        blockAlign: view.getUint16(body + 12, true),
        bits: view.getUint16(body + 14, true),
      };
      if (format.tag === 0xfffe && size >= 40) format.tag = view.getUint16(body + 24, true);
    } else if (id === "data") {
      dataOffset = body;
      dataLength = size;
    }
    offset = body + size + (size & 1);
  }

  assert(format && dataOffset >= 0, "The WAV file has no audio data.");
  assert(format.channels > 0 && format.blockAlign > 0, "The WAV format is invalid.");
  assert(format.tag === 1 || format.tag === 3, "Only PCM and IEEE-float WAV files are supported.");

  const bytesPerSample = format.bits / 8;
  assert(Number.isInteger(bytesPerSample) && [1, 2, 3, 4, 8].includes(bytesPerSample), `Unsupported ${format.bits}-bit WAV.`);
  const frameCount = Math.floor(dataLength / format.blockAlign);
  const samples = new Float64Array(frameCount);
  for (let frame = 0; frame < frameCount; frame++) {
    const offset = dataOffset + frame * format.blockAlign;
    if (format.tag === 3 && format.bits === 32) samples[frame] = view.getFloat32(offset, true);
    else if (format.tag === 3 && format.bits === 64) samples[frame] = view.getFloat64(offset, true);
    else if (format.tag === 1 && format.bits === 8) samples[frame] = (view.getUint8(offset) - 128) / 128;
    else if (format.tag === 1 && format.bits === 16) samples[frame] = view.getInt16(offset, true) / 32768;
    else if (format.tag === 1 && format.bits === 24) samples[frame] = readInt24(view, offset) / 8388608;
    else if (format.tag === 1 && format.bits === 32) samples[frame] = view.getInt32(offset, true) / 2147483648;
    else throw new Error(`Unsupported WAV encoding: format ${format.tag}, ${format.bits} bits.`);
  }
  return { samples, sampleRate: format.sampleRate };
}

function readAscii(view, offset, length) {
  let value = "";
  for (let i = 0; i < length; i++) value += String.fromCharCode(view.getUint8(offset + i));
  return value;
}

function readInt24(view, offset) {
  const value = view.getUint8(offset) | (view.getUint8(offset + 1) << 8) | (view.getUint8(offset + 2) << 16);
  return value & 0x800000 ? value | 0xff000000 : value;
}

export function encodePcmWav(samples, sampleRate, bits = 16) {
  assert(bits === 16 || bits === 24, "Output must be 16-bit or 24-bit PCM.");
  const bytesPerSample = bits / 8;
  const dataLength = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeAscii(view, 8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, bits, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLength, true);

  const scale = bits === 16 ? 32768 : 8388608;
  const min = -scale;
  const max = scale - 1;
  for (let i = 0; i < samples.length; i++) {
    // libsndfile truncates normalized floating-point PCM toward negative infinity.
    const value = Math.max(min, Math.min(max, Math.floor(Math.fround(samples[i]) * scale)));
    const offset = 44 + i * bytesPerSample;
    if (bits === 16) view.setInt16(offset, value, true);
    else {
      view.setUint8(offset, value & 0xff);
      view.setUint8(offset + 1, (value >>> 8) & 0xff);
      view.setUint8(offset + 2, (value >>> 16) & 0xff);
    }
  }
  return buffer;
}

export function createStoredZip(files) {
  const encoder = new TextEncoder();
  const entries = files.map(({ name, data }) => {
    const nameBytes = encoder.encode(name);
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    return { nameBytes, bytes, crc: crc32(bytes), offset: 0 };
  });
  const localSize = entries.reduce((sum, entry) => sum + 30 + entry.nameBytes.length + entry.bytes.length, 0);
  const centralSize = entries.reduce((sum, entry) => sum + 46 + entry.nameBytes.length, 0);
  const result = new Uint8Array(localSize + centralSize + 22);
  const view = new DataView(result.buffer);
  let offset = 0;
  for (const entry of entries) {
    entry.offset = offset;
    view.setUint32(offset, 0x04034b50, true);
    view.setUint16(offset + 4, 20, true);
    view.setUint16(offset + 6, 0, true);
    view.setUint16(offset + 8, 0, true);
    view.setUint16(offset + 10, 0, true);
    view.setUint16(offset + 12, 0, true);
    view.setUint32(offset + 14, entry.crc, true);
    view.setUint32(offset + 18, entry.bytes.length, true);
    view.setUint32(offset + 22, entry.bytes.length, true);
    view.setUint16(offset + 26, entry.nameBytes.length, true);
    view.setUint16(offset + 28, 0, true);
    result.set(entry.nameBytes, offset + 30);
    result.set(entry.bytes, offset + 30 + entry.nameBytes.length);
    offset += 30 + entry.nameBytes.length + entry.bytes.length;
  }
  const centralOffset = offset;
  for (const entry of entries) {
    view.setUint32(offset, 0x02014b50, true);
    view.setUint16(offset + 4, 20, true);
    view.setUint16(offset + 6, 20, true);
    view.setUint16(offset + 8, 0, true);
    view.setUint16(offset + 10, 0, true);
    view.setUint16(offset + 12, 0, true);
    view.setUint16(offset + 14, 0, true);
    view.setUint32(offset + 16, entry.crc, true);
    view.setUint32(offset + 20, entry.bytes.length, true);
    view.setUint32(offset + 24, entry.bytes.length, true);
    view.setUint16(offset + 28, entry.nameBytes.length, true);
    view.setUint16(offset + 30, 0, true);
    view.setUint16(offset + 32, 0, true);
    view.setUint16(offset + 34, 0, true);
    view.setUint16(offset + 36, 0, true);
    view.setUint32(offset + 38, 0, true);
    view.setUint32(offset + 42, entry.offset, true);
    result.set(entry.nameBytes, offset + 46);
    offset += 46 + entry.nameBytes.length;
  }
  view.setUint32(offset, 0x06054b50, true);
  view.setUint16(offset + 4, 0, true);
  view.setUint16(offset + 6, 0, true);
  view.setUint16(offset + 8, entries.length, true);
  view.setUint16(offset + 10, entries.length, true);
  view.setUint32(offset + 12, centralSize, true);
  view.setUint32(offset + 16, centralOffset, true);
  view.setUint16(offset + 20, 0, true);
  return result.buffer;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function writeAscii(view, offset, value) {
  for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
}

export function generateStandardIr(piezo, mic, sampleRate, irLength = 2048, smoothing = 1 / 3, progress = () => {}) {
  assert(piezo.length >= sampleRate * 2 && mic.length >= sampleRate * 2, "Audio files are too short. At least 2 seconds are required.");
  assert(isPowerOfTwo(irLength), "IR length must be a power of two.");
  const frameCount = Math.min(piezo.length, mic.length);
  const chunkCount = Math.floor(frameCount / sampleRate) - 1;
  const transfers = Array.from({ length: chunkCount }, () => ({ re: new Float64Array(irLength), im: new Float64Array(irLength) }));

  progress("Extracting transfer functions…", 10);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const start = sampleRate + chunk * sampleRate;
    const pRe = new Float64Array(irLength);
    const mRe = new Float64Array(irLength);
    pRe.set(piezo.subarray(start, Math.min(start + irLength, piezo.length)));
    mRe.set(mic.subarray(start, Math.min(start + irLength, mic.length)));
    const pIm = new Float64Array(irLength);
    const mIm = new Float64Array(irLength);
    fftInPlace(pRe, pIm, false);
    fftInPlace(mRe, mIm, false);
    const transfer = transfers[chunk];
    for (let bin = 0; bin < irLength; bin++) {
      const pr = pRe[bin] + 1e-12;
      const pi = pIm[bin];
      const denominator = pr * pr + pi * pi;
      transfer.re[bin] = (mRe[bin] * pr + mIm[bin] * pi) / denominator;
      transfer.im[bin] = (mIm[bin] * pr - mRe[bin] * pi) / denominator;
    }
    if (chunk % 5 === 0) progress(`Extracting chunk ${chunk + 1}/${chunkCount}…`, 10 + Math.trunc((chunk / chunkCount) * 20));
  }

  progress("Rejecting outliers and averaging…", 30);
  const averageRe = new Float64Array(irLength);
  const averageIm = new Float64Array(irLength);
  for (let bin = 0; bin < irLength; bin++) {
    let meanRe = 0;
    let meanIm = 0;
    for (const transfer of transfers) {
      meanRe += transfer.re[bin];
      meanIm += transfer.im[bin];
    }
    meanRe /= chunkCount;
    meanIm /= chunkCount;
    let variance = 0;
    for (const transfer of transfers) {
      const dr = transfer.re[bin] - meanRe;
      const di = transfer.im[bin] - meanIm;
      variance += dr * dr + di * di;
    }
    const std = Math.sqrt(variance / chunkCount);
    let keptRe = 0;
    let keptIm = 0;
    let kept = 0;
    for (const transfer of transfers) {
      const magnitude = Math.hypot(transfer.re[bin], transfer.im[bin]);
      if (std === 0 || Math.hypot(magnitude - meanRe, -meanIm) < 2 * std) {
        keptRe += transfer.re[bin];
        keptIm += transfer.im[bin];
        kept++;
      }
    }
    if (kept) {
      averageRe[bin] = keptRe / kept;
      averageIm[bin] = keptIm / kept;
    } else averageRe[bin] = 1;
  }

  if (smoothing > 0) octaveSmoothInPlace(averageRe, averageIm, smoothing);
  fftInPlace(averageRe, averageIm, true);
  for (let i = 0; i < irLength; i++) {
    const n = i + irLength - 1;
    averageRe[i] *= 0.42 - 0.5 * Math.cos((TWO_PI * n) / (2 * irLength - 1)) + 0.08 * Math.cos((2 * TWO_PI * n) / (2 * irLength - 1));
    averageIm[i] = 0;
  }

  progress("Converting to minimum phase…", 50);
  const ir = minimumPhase(averageRe);
  const fadeLength = Math.trunc(irLength * 0.1);
  for (let i = 0; i < fadeLength; i++) ir[irLength - fadeLength + i] *= 1 - i / (fadeLength - 1);
  normalizeInPlace(ir, 0.95);

  progress("Measuring one-third-octave response…", 60);
  const testStart = Math.min(10 * sampleRate, Math.trunc(mic.length / 3));
  const testEnd = Math.min(20 * sampleRate, Math.trunc(mic.length / 3) * 2);
  const micSection = mic.slice(testStart, testEnd);
  const piezoSection = piezo.slice(testStart, testEnd);
  const simulated = convolveSame(piezoSection, ir);
  normalizeInPlace(micSection, 1, 1e-12);
  normalizeInPlace(simulated, 1, 1e-12);
  const micSpectrum = thirdOctaveSpectrum(micSection, sampleRate);
  const piezoSpectrum = thirdOctaveSpectrum(simulated, sampleRate);

  let corrected = ir.slice();
  for (let band = 0; band < micSpectrum.lower.length; band++) {
    progress(`Matching EQ band ${band + 1}/${micSpectrum.lower.length}…`, 60 + Math.trunc((band / micSpectrum.lower.length) * 30));
    const gainDb = clamp(micSpectrum.levels[band] - piezoSpectrum.levels[band], -24, 24);
    const gain = 10 ** (gainDb / 20) - 1;
    const [b, a] = butterworthBandpass(micSpectrum.lower[band], micSpectrum.upper[band], sampleRate);
    const filtered = lfilter(b, a, corrected);
    for (let i = 0; i < corrected.length; i++) corrected[i] += filtered[i] * gain;
  }
  normalizeInPlace(corrected, 0.99);
  progress("Encoding WAV…", 100);
  return corrected;
}

export function generateCukiIrs(piezo, mic, sampleRate, irLength = 2048, progress = () => {}) {
  assert(isPowerOfTwo(irLength), "IR length must be a power of two.");
  const frameCount = Math.min(piezo.length, mic.length);
  const chunkCount = Math.floor(frameCount / sampleRate) - 10;
  assert(chunkCount > 0, "Cuki's algorithm requires at least 11 seconds of audio.");
  const transfers = Array.from({ length: chunkCount }, () => ({ re: new Float64Array(irLength), im: new Float64Array(irLength) }));

  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const start = 3 * sampleRate + chunk * sampleRate;
    const pRe = new Float64Array(irLength);
    const mRe = new Float64Array(irLength);
    pRe.set(piezo.subarray(start, Math.min(start + irLength, piezo.length)));
    mRe.set(mic.subarray(start, Math.min(start + irLength, mic.length)));
    const pIm = new Float64Array(irLength);
    const mIm = new Float64Array(irLength);
    fftInPlace(pRe, pIm, false);
    fftInPlace(mRe, mIm, false);
    const transfer = transfers[chunk];
    let valid = true;
    for (let bin = 0; bin < irLength; bin++) {
      const denominator = pRe[bin] * pRe[bin] + pIm[bin] * pIm[bin];
      transfer.re[bin] = (mRe[bin] * pRe[bin] + mIm[bin] * pIm[bin]) / denominator;
      transfer.im[bin] = (mIm[bin] * pRe[bin] - mRe[bin] * pIm[bin]) / denominator;
      valid &&= Number.isFinite(transfer.re[bin]) && Number.isFinite(transfer.im[bin]);
    }
    if (!valid) {
      transfer.re.fill(1);
      transfer.im.fill(0);
    }
    if (chunk % 5 === 0 || chunk === chunkCount - 1) progress(`Extracting chunk ${chunk + 1}/${chunkCount}…`, 10 + Math.trunc((chunk / chunkCount) * 20));
  }

  const averageRe = new Float64Array(irLength);
  const averageIm = new Float64Array(irLength);
  robustAverage(transfers, averageRe, averageIm, progress);
  fftInPlace(averageRe, averageIm, true);
  for (let i = 0; i < irLength; i++) {
    const n = i + irLength - 1;
    const window = 0.42 - 0.5 * Math.cos((TWO_PI * n) / (2 * irLength - 1)) + 0.08 * Math.cos((2 * TWO_PI * n) / (2 * irLength - 1));
    averageRe[i] *= window;
    averageIm[i] *= window;
  }
  let maximum = 0;
  for (let i = 0; i < irLength; i++) maximum = Math.max(maximum, Math.hypot(averageRe[i], averageIm[i]));
  const standard = new Float64Array(irLength);
  for (let i = 0; i < irLength; i++) standard[i] = (averageRe[i] / maximum) * 0.95;

  const start = 10 * sampleRate + 1;
  assert(frameCount >= 20 * sampleRate, "Cuki's algorithm requires at least 20 seconds of audio for EQ matching.");
  const end = 20 * sampleRate;
  const micSection = mic.slice(start, end);
  const piezoSection = piezo.slice(start, end);
  const simulated = convolveSame(piezoSection, standard);
  normalizeInPlace(micSection, 1);
  normalizeInPlace(simulated, 1);
  const micSpectrum = thirdOctaveSpectrum(micSection, sampleRate, false);
  const piezoSpectrum = thirdOctaveSpectrum(simulated, sampleRate, false);

  let matched = standard.slice();
  for (let band = 0; band < micSpectrum.lower.length; band++) {
    progress(`Matching EQ band ${band + 1}/${micSpectrum.lower.length}…`, 60 + Math.trunc((band / micSpectrum.lower.length) * 30));
    const gain = 10 ** ((micSpectrum.levels[band] - piezoSpectrum.levels[band]) / 20) - 1;
    const [b, a] = butterworthBandpass(micSpectrum.lower[band], micSpectrum.upper[band], sampleRate);
    const filtered = lfilter(b, a, matched);
    for (let i = 0; i < matched.length; i++) matched[i] += filtered[i] * gain;
  }
  normalizeInPlace(matched, 0.95);
  const blended = standard.slice();
  blended[0] += 1;
  for (let i = 0; i < blended.length; i++) blended[i] /= 2;
  progress("Encoding WAVs…", 100);
  return { matched, standard, blended };
}

export function generateComparisonSpectrum(piezo, mic, ir, sampleRate, progress = () => {}) {
  progress("Applying IR for comparison plot…", 93);
  const segmentLength = Math.min(8192, mic.length);
  assert(isPowerOfTwo(segmentLength), "Comparison FFT length must be a power of two.");
  const processedPickup = convolveFull(piezo, ir);
  progress("Generating comparison plot…", 96);
  const targetPower = welchPower(mic, sampleRate, segmentLength);
  const resultPower = welchPower(processedPickup, sampleRate, segmentLength);

  const frequencies = new Float64Array(targetPower.length);
  const targetDb = new Float64Array(targetPower.length);
  const resultDb = new Float64Array(targetPower.length);
  for (let bin = 0; bin < targetPower.length; bin++) {
    frequencies[bin] = (bin * sampleRate) / segmentLength;
    targetDb[bin] = 10 * Math.log10(targetPower[bin] + 1e-12);
    resultDb[bin] = 10 * Math.log10(resultPower[bin] + 1e-12);
  }
  return { frequencies, targetDb, resultDb };
}

function welchPower(samples, sampleRate, segmentLength) {
  const step = Math.trunc(segmentLength / 2);
  const bins = Math.trunc(segmentLength / 2) + 1;
  const power = new Float64Array(bins);
  const window = new Float64Array(segmentLength);
  let windowPower = 0;
  for (let i = 0; i < segmentLength; i++) {
    window[i] = 0.5 - 0.5 * Math.cos((TWO_PI * i) / segmentLength);
    windowPower += window[i] * window[i];
  }

  let segments = 0;
  for (let start = 0; start + segmentLength <= samples.length; start += step) {
    let mean = 0;
    for (let i = 0; i < segmentLength; i++) mean += samples[start + i];
    mean /= segmentLength;
    const re = new Float64Array(segmentLength);
    const im = new Float64Array(segmentLength);
    for (let i = 0; i < segmentLength; i++) re[i] = (samples[start + i] - mean) * window[i];
    fftInPlace(re, im);
    for (let bin = 0; bin < bins; bin++) {
      const oneSidedScale = bin === 0 || bin === bins - 1 ? 1 : 2;
      power[bin] += oneSidedScale * (re[bin] * re[bin] + im[bin] * im[bin]);
    }
    segments++;
  }
  const scale = 1 / (segments * sampleRate * windowPower);
  for (let bin = 0; bin < bins; bin++) power[bin] *= scale;
  return power;
}

function robustAverage(transfers, outputRe, outputIm, progress = () => {}) {
  const count = transfers.length;
  for (let bin = 0; bin < outputRe.length; bin++) {
    let meanRe = 0;
    let meanIm = 0;
    for (const transfer of transfers) {
      meanRe += transfer.re[bin];
      meanIm += transfer.im[bin];
    }
    meanRe /= count;
    meanIm /= count;
    let variance = 0;
    for (const transfer of transfers) {
      const dr = transfer.re[bin] - meanRe;
      const di = transfer.im[bin] - meanIm;
      variance += dr * dr + di * di;
    }
    const std = Math.sqrt(variance / count);
    let keptRe = 0;
    let keptIm = 0;
    let kept = 0;
    for (const transfer of transfers) {
      if (Math.hypot(Math.hypot(transfer.re[bin], transfer.im[bin]) - meanRe, -meanIm) < 2 * std) {
        keptRe += transfer.re[bin];
        keptIm += transfer.im[bin];
        kept++;
      }
    }
    if (kept) {
      outputRe[bin] = keptRe / kept;
      outputIm[bin] = keptIm / kept;
    } else outputRe[bin] = 1;
    if (bin % 500 === 0 || bin === outputRe.length - 1) progress(`Averaging frequency ${bin + 1}/${outputRe.length}…`, 30 + Math.trunc((bin / outputRe.length) * 20));
  }
}

function octaveSmoothInPlace(re, im, fraction) {
  const magnitude = new Float64Array(re.length);
  for (let i = 0; i < re.length; i++) magnitude[i] = Math.hypot(re[i], im[i]);
  const smoothed = magnitude.slice();
  const widthFactor = 2 ** (fraction / 2) - 2 ** (-fraction / 2);
  for (let i = 1; i < re.length; i++) {
    const width = Math.trunc(i * widthFactor);
    if (width > 1) {
      const start = Math.max(0, i - Math.trunc(width / 2));
      const end = Math.min(re.length, i + Math.trunc(width / 2));
      let total = 0;
      for (let j = start; j < end; j++) total += magnitude[j];
      smoothed[i] = total / (end - start);
    }
  }
  for (let i = 0; i < re.length; i++) {
    const phase = Math.atan2(im[i], re[i]);
    re[i] = smoothed[i] * Math.cos(phase);
    im[i] = smoothed[i] * Math.sin(phase);
  }
}

function minimumPhase(input) {
  const n = input.length;
  const re = input.slice();
  const im = new Float64Array(n);
  fftInPlace(re, im, false);
  for (let i = 0; i < n; i++) {
    re[i] = Math.log(Math.hypot(re[i], im[i]) + 1e-10);
    im[i] = 0;
  }
  fftInPlace(re, im, true);
  for (let i = 1; i < n / 2; i++) re[i] *= 2;
  for (let i = n / 2 + 1; i < n; i++) re[i] = 0;
  im.fill(0);
  fftInPlace(re, im, false);
  for (let i = 0; i < n; i++) {
    const magnitude = Math.exp(re[i]);
    const phase = im[i];
    re[i] = magnitude * Math.cos(phase);
    im[i] = magnitude * Math.sin(phase);
  }
  fftInPlace(re, im, true);
  return re;
}

function thirdOctaveSpectrum(input, sampleRate, addEpsilon = true) {
  const G = 10 ** (3 / 10);
  const levels = [];
  const lower = [];
  const upper = [];
  for (let x = 1; x <= 42; x++) {
    const center = 1000 * G ** ((x - 30) / 3);
    const high = center * G ** (1 / 6);
    if (high < 20 || high > sampleRate / 2) continue;
    const low = center * G ** (-1 / 6);
    const [b, a] = butterworthBandpass(low, high, sampleRate);
    const filtered = lfilter(b, a, input);
    let power = 0;
    for (let i = 0; i < filtered.length; i++) power += filtered[i] * filtered[i];
    levels.push(10 * Math.log10(power / filtered.length + (addEpsilon ? 1e-12 : 0)));
    lower.push(low);
    upper.push(high);
  }
  return { levels, lower, upper };
}

function butterworthBandpass(low, high, sampleRate) {
  const warpedLow = 2 * sampleRate * Math.tan((Math.PI * low) / sampleRate);
  const warpedHigh = 2 * sampleRate * Math.tan((Math.PI * high) / sampleRate);
  const bandwidth = warpedHigh - warpedLow;
  const center = Math.sqrt(warpedLow * warpedHigh);
  const prototype = [complex(-Math.SQRT1_2, Math.SQRT1_2), complex(-Math.SQRT1_2, -Math.SQRT1_2)];
  const analogPoles = [];
  for (const pole of prototype) {
    const scaled = mulReal(pole, bandwidth / 2);
    const root = sqrtComplex(sub(mul(scaled, scaled), complex(center * center, 0)));
    analogPoles.push(add(scaled, root), sub(scaled, root));
  }
  const fs2 = 2 * sampleRate;
  const digitalPoles = analogPoles.map((pole) => div(add(complex(fs2, 0), pole), sub(complex(fs2, 0), pole)));
  const digitalZeros = [complex(1, 0), complex(1, 0), complex(-1, 0), complex(-1, 0)];
  let denominatorGain = complex(1, 0);
  for (const pole of analogPoles) denominatorGain = mul(denominatorGain, sub(complex(fs2, 0), pole));
  const gain = (bandwidth * bandwidth * fs2 * fs2) / denominatorGain.re;
  const b = polynomialFromRoots(digitalZeros).map((v) => v * gain);
  const a = polynomialFromRoots(digitalPoles);
  const a0 = a[0];
  return [b.map((v) => v / a0), a.map((v) => v / a0)];
}

function polynomialFromRoots(roots) {
  let coefficients = [complex(1, 0)];
  for (const root of roots) {
    const next = Array.from({ length: coefficients.length + 1 }, () => complex(0, 0));
    for (let i = 0; i < coefficients.length; i++) {
      next[i] = add(next[i], coefficients[i]);
      next[i + 1] = sub(next[i + 1], mul(coefficients[i], root));
    }
    coefficients = next;
  }
  return coefficients.map((value) => value.re);
}

function lfilter(b, a, input) {
  const output = new Float64Array(input.length);
  const state = new Float64Array(Math.max(a.length, b.length) - 1);
  for (let n = 0; n < input.length; n++) {
    const value = b[0] * input[n] + state[0];
    output[n] = value;
    for (let i = 0; i < state.length - 1; i++) state[i] = (b[i + 1] || 0) * input[n] + state[i + 1] - (a[i + 1] || 0) * value;
    state[state.length - 1] = (b[state.length] || 0) * input[n] - (a[state.length] || 0) * value;
  }
  return output;
}

function convolveSame(left, right) {
  const fullLength = left.length + right.length - 1;
  const fftLength = nextPowerOfTwo(fullLength);
  const leftRe = new Float64Array(fftLength);
  const rightRe = new Float64Array(fftLength);
  leftRe.set(left);
  rightRe.set(right);
  const leftIm = new Float64Array(fftLength);
  const rightIm = new Float64Array(fftLength);
  fftInPlace(leftRe, leftIm, false);
  fftInPlace(rightRe, rightIm, false);
  for (let i = 0; i < fftLength; i++) {
    const re = leftRe[i] * rightRe[i] - leftIm[i] * rightIm[i];
    leftIm[i] = leftRe[i] * rightIm[i] + leftIm[i] * rightRe[i];
    leftRe[i] = re;
  }
  fftInPlace(leftRe, leftIm, true);
  const start = Math.trunc((right.length - 1) / 2);
  return leftRe.slice(start, start + left.length);
}

function convolveFull(left, right) {
  const blockLength = 8192;
  const fftLength = nextPowerOfTwo(blockLength + right.length - 1);
  const kernelRe = new Float64Array(fftLength);
  const kernelIm = new Float64Array(fftLength);
  kernelRe.set(right);
  fftInPlace(kernelRe, kernelIm);

  const output = new Float64Array(left.length + right.length - 1);
  const re = new Float64Array(fftLength);
  const im = new Float64Array(fftLength);
  for (let offset = 0; offset < left.length; offset += blockLength) {
    re.fill(0);
    im.fill(0);
    const count = Math.min(blockLength, left.length - offset);
    re.set(left.subarray(offset, offset + count));
    fftInPlace(re, im);
    for (let bin = 0; bin < fftLength; bin++) {
      const productRe = re[bin] * kernelRe[bin] - im[bin] * kernelIm[bin];
      im[bin] = re[bin] * kernelIm[bin] + im[bin] * kernelRe[bin];
      re[bin] = productRe;
    }
    fftInPlace(re, im, true);
    const valid = count + right.length - 1;
    for (let i = 0; i < valid; i++) output[offset + i] += re[i];
  }
  return output;
}

export function fftInPlace(re, im, inverse = false) {
  const n = re.length;
  assert(n === im.length && isPowerOfTwo(n), "FFT length must be a power of two.");
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = (inverse ? TWO_PI : -TWO_PI) / length;
    const stepRe = Math.cos(angle);
    const stepIm = Math.sin(angle);
    for (let offset = 0; offset < n; offset += length) {
      let twiddleRe = 1;
      let twiddleIm = 0;
      for (let j = 0; j < length / 2; j++) {
        const even = offset + j;
        const odd = even + length / 2;
        const oddRe = re[odd] * twiddleRe - im[odd] * twiddleIm;
        const oddIm = re[odd] * twiddleIm + im[odd] * twiddleRe;
        re[odd] = re[even] - oddRe;
        im[odd] = im[even] - oddIm;
        re[even] += oddRe;
        im[even] += oddIm;
        const nextRe = twiddleRe * stepRe - twiddleIm * stepIm;
        twiddleIm = twiddleRe * stepIm + twiddleIm * stepRe;
        twiddleRe = nextRe;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function normalizeInPlace(samples, peak, epsilon = 0) {
  let maximum = 0;
  for (let i = 0; i < samples.length; i++) maximum = Math.max(maximum, Math.abs(samples[i]));
  const scale = peak / (maximum + epsilon);
  for (let i = 0; i < samples.length; i++) samples[i] *= scale;
}

function isPowerOfTwo(value) { return value > 0 && (value & (value - 1)) === 0; }
function nextPowerOfTwo(value) { return 2 ** Math.ceil(Math.log2(value)); }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function complex(re, im) { return { re, im }; }
function add(a, b) { return complex(a.re + b.re, a.im + b.im); }
function sub(a, b) { return complex(a.re - b.re, a.im - b.im); }
function mul(a, b) { return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
function mulReal(a, b) { return complex(a.re * b, a.im * b); }
function div(a, b) { const d = b.re * b.re + b.im * b.im; return complex((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d); }
function sqrtComplex(value) {
  const magnitude = Math.hypot(value.re, value.im);
  const re = Math.sqrt((magnitude + value.re) / 2);
  return complex(re, Math.sign(value.im || 1) * Math.sqrt((magnitude - value.re) / 2));
}
