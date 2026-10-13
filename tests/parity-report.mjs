import { readFile } from "node:fs/promises";
import { decodeWav } from "../static/dsp.js";

if (process.argv.length !== 4) {
  console.error("usage: node tests/parity-report.mjs REFERENCE.wav ACTUAL.wav");
  process.exit(2);
}

const load = async (path) => {
  const bytes = await readFile(path);
  return { bytes, wav: decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)) };
};
const [reference, actual] = await Promise.all(process.argv.slice(2).map(load));
if (reference.wav.samples.length !== actual.wav.samples.length) throw new Error("Sample counts differ.");
let differentSamples = 0;
let maximumDelta = 0;
let squaredError = 0;
for (let i = 0; i < reference.wav.samples.length; i++) {
  const delta = Math.abs(reference.wav.samples[i] - actual.wav.samples[i]);
  if (delta) differentSamples++;
  maximumDelta = Math.max(maximumDelta, delta);
  squaredError += delta * delta;
}
console.log(JSON.stringify({
  byteIdentical: reference.bytes.equals(actual.bytes),
  sampleCount: reference.wav.samples.length,
  differentSamples,
  maximumDelta,
  rmsError: Math.sqrt(squaredError / reference.wav.samples.length),
}, null, 2));
