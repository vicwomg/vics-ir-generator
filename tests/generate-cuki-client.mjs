import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { createStoredZip, decodeWav, encodePcmWav, generateCukiIrs } from "../static/dsp.js";

if (process.argv.length !== 5) {
  console.error("usage: node tests/generate-cuki-client.mjs PIEZO.wav MIC.wav OUTPUT_DIR");
  process.exit(2);
}

const readArrayBuffer = async (path) => {
  const data = await readFile(path);
  return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
};

const piezo = decodeWav(await readArrayBuffer(process.argv[2]));
const mic = decodeWav(await readArrayBuffer(process.argv[3]));
if (piezo.sampleRate !== mic.sampleRate) throw new Error("Sample rates do not match.");
const result = generateCukiIrs(piezo.samples, mic.samples, piezo.sampleRate, 2048);
const prefix = `IR_cuki_${String(piezo.sampleRate / 1000).slice(0, 2)}k_2048`;
await mkdir(process.argv[4], { recursive: true });
const files = [];
for (const [suffix, samples] of [["M", result.matched], ["Std", result.standard], ["Std_Bld", result.blended]]) {
  const name = `${prefix}_${suffix}.wav`;
  const data = encodePcmWav(samples, piezo.sampleRate, 24);
  const path = join(process.argv[4], name);
  await writeFile(path, new Uint8Array(data));
  files.push({ name, data });
  console.error(`wrote ${basename(path)}`);
}
await writeFile(join(process.argv[4], "cuki_irs.zip"), new Uint8Array(createStoredZip(files)));
