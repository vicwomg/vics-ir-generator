import { readFile, writeFile } from "node:fs/promises";
import { decodeWav, encodePcmWav, generateStandardIr } from "../static/dsp.js";

if (process.argv.length !== 5) {
  console.error("usage: node tests/generate-client.mjs PIEZO.wav MIC.wav OUTPUT.wav");
  process.exit(2);
}

const readArrayBuffer = async (path) => {
  const data = await readFile(path);
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
};

const piezo = decodeWav(await readArrayBuffer(process.argv[2]));
const mic = decodeWav(await readArrayBuffer(process.argv[3]));
if (piezo.sampleRate !== mic.sampleRate) throw new Error("Sample rates do not match.");

const ir = generateStandardIr(piezo.samples, mic.samples, piezo.sampleRate, 2048, 0.33333333, (message, percent) => {
  if (percent % 10 === 0) console.error(`${percent}% ${message}`);
});
if (process.env.PARITY_INDICES) {
  console.error(process.env.PARITY_INDICES.split(",").map((value) => {
    const index = Number(value);
    const sample = Math.fround(ir[index]);
    return [index, sample, sample * 32768];
  }));
}
await writeFile(process.argv[4], new Uint8Array(encodePcmWav(ir, piezo.sampleRate)));
