import assert from "node:assert/strict";
import test from "node:test";
import { createStoredZip, decodeWav, encodePcmWav, fftInPlace, generateComparisonSpectrum, generateStandardIr } from "../static/dsp.js";

test("16-bit WAV encoding round-trips mono samples and metadata", () => {
  const input = new Float64Array([-1, -0.5, 0, 0.5, 0.9999]);
  const encoded = encodePcmWav(input, 44100, 16);
  const decoded = decodeWav(encoded);
  assert.equal(decoded.sampleRate, 44100);
  assert.equal(decoded.samples.length, input.length);
  for (let i = 0; i < input.length; i++) assert.ok(Math.abs(decoded.samples[i] - input[i]) <= 1 / 32768);
});

test("24-bit WAV encoding round-trips mono samples", () => {
  const input = new Float64Array([-0.9, -0.1, 0.1, 0.9]);
  const decoded = decodeWav(encodePcmWav(input, 48000, 24));
  for (let i = 0; i < input.length; i++) assert.ok(Math.abs(decoded.samples[i] - input[i]) <= 1 / 8388608);
});

test("WAV encoding rejects non-finite samples", () => {
  assert.throws(() => encodePcmWav(new Float64Array([0, Number.NaN]), 44100), /non-finite samples/);
  assert.throws(() => encodePcmWav(new Float64Array([Number.POSITIVE_INFINITY]), 44100, 24), /non-finite samples/);
});

test("FFT inverse reconstructs its input", () => {
  const original = Float64Array.from({ length: 2048 }, (_, i) => Math.sin(i * 0.17) + Math.cos(i * 0.031));
  const re = original.slice();
  const im = new Float64Array(re.length);
  fftInPlace(re, im);
  fftInPlace(re, im, true);
  for (let i = 0; i < re.length; i++) assert.ok(Math.abs(re[i] - original[i]) < 1e-12);
});

test("stored ZIP contains valid local and central directory records", () => {
  const zip = new Uint8Array(createStoredZip([{ name: "ir.wav", data: new Uint8Array([1, 2, 3]) }]));
  const view = new DataView(zip.buffer);
  assert.equal(view.getUint32(0, true), 0x04034b50);
  assert.equal(view.getUint32(zip.length - 22, true), 0x06054b50);
  assert.equal(view.getUint16(zip.length - 12, true), 1);
});

test("comparison plot shows identical spectra for an identity IR", () => {
  const sampleRate = 44100;
  const samples = Float64Array.from({ length: 16384 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / sampleRate));
  const plot = generateComparisonSpectrum(samples, samples, new Float64Array([1]), sampleRate);
  assert.equal(plot.frequencies.length, 4097);
  for (let i = 0; i < plot.targetDb.length; i++) assert.ok(Math.abs(plot.targetDb[i] - plot.resultDb[i]) < 1e-8);
});

test("comparison plot supports non-power-of-two shared extents", () => {
  const sampleRate = 4000;
  const samples = Float64Array.from({ length: 8000 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / sampleRate));
  const plot = generateComparisonSpectrum(samples, samples, new Float64Array([1]), sampleRate);
  assert.equal(plot.frequencies.length, 2049);
  for (let i = 0; i < plot.targetDb.length; i++) assert.ok(Math.abs(plot.targetDb[i] - plot.resultDb[i]) < 1e-8);
});

test("comparison plot rejects inputs too short for Welch analysis", () => {
  const samples = new Float64Array([1]);
  assert.throws(() => generateComparisonSpectrum(samples, samples, new Float64Array([1]), 4000), /at least 2 aligned audio samples/);
});

test("comparison plot uses the shared extent of unequal input lengths", () => {
  const sampleRate = 44100;
  const pickup = Float64Array.from({ length: 16384 }, (_, i) => Math.sin((2 * Math.PI * 440 * i) / sampleRate));
  const mic = Float64Array.from({ length: 32768 }, (_, i) => (
    i < pickup.length ? pickup[i] : Math.sin((2 * Math.PI * 1200 * i) / sampleRate)
  ));
  const plot = generateComparisonSpectrum(pickup, mic, new Float64Array([1]), sampleRate);
  for (let i = 0; i < plot.targetDb.length; i++) assert.ok(Math.abs(plot.targetDb[i] - plot.resultDb[i]) < 1e-8);
});

test("comparison plot analyzes the convolved pickup instead of multiplying spectra", () => {
  const sampleRate = 44100;
  const pickup = Float64Array.from({ length: 16384 }, (_, i) => (
    Math.sin((2 * Math.PI * 53 * i) / sampleRate) * (i < 9000 ? 0.8 : 0.2)
    + Math.sin((2 * Math.PI * 1200 * i) / sampleRate) * 0.1
  ));
  const ir = Float64Array.from({ length: 64 }, (_, i) => Math.exp(-i / 12) * Math.cos(i * 0.4));
  const expected = new Float64Array(pickup.length + ir.length - 1);
  for (let i = 0; i < pickup.length; i++) {
    for (let j = 0; j < ir.length; j++) expected[i + j] += pickup[i] * ir[j];
  }
  const plot = generateComparisonSpectrum(pickup, expected, ir, sampleRate);
  for (let i = 0; i < plot.targetDb.length; i++) assert.ok(Math.abs(plot.targetDb[i] - plot.resultDb[i]) < 1e-8);
});

test("standard generation uses the shared extent of mismatched input lengths", () => {
  const sampleRate = 2048;
  const pickup = Float64Array.from({ length: sampleRate * 2 }, (_, i) => (
    Math.sin((2 * Math.PI * 110 * i) / sampleRate) + 0.25 * Math.sin((2 * Math.PI * 370 * i) / sampleRate)
  ));
  const mic = Float64Array.from({ length: sampleRate * 6 }, (_, i) => (
    Math.sin((2 * Math.PI * 110 * i) / sampleRate) + 0.25 * Math.sin((2 * Math.PI * 370 * i) / sampleRate)
  ));
  const ir = generateStandardIr(pickup, mic, sampleRate, 64, 0);
  assert.equal(ir.length, 64);
  assert.ok(ir.every(Number.isFinite));
  assert.ok(ir.some((sample) => sample !== 0));
});

test("standard generation rejects an unavailable EQ interval", () => {
  const samples = new Float64Array([1]);
  assert.throws(() => generateStandardIr(samples, samples, 0, 64, 0), /usable EQ matching interval/);
});
