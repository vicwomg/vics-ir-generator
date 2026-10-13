import { readFile, writeFile } from "node:fs/promises";
import { decodeWav, generateComparisonSpectrum } from "../static/dsp.js";

if (process.argv.length !== 6) {
  console.error("usage: node tests/generate-comparison-data.mjs PIEZO.wav MIC.wav IR.wav OUTPUT.json");
  process.exit(2);
}

const load = async (path) => {
  const bytes = await readFile(path);
  return decodeWav(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const [piezo, mic, ir] = await Promise.all(process.argv.slice(2, 5).map(load));
const plot = generateComparisonSpectrum(piezo.samples, mic.samples, ir.samples, piezo.sampleRate);
await writeFile(process.argv[5], JSON.stringify({
  frequencies: Array.from(plot.frequencies),
  targetDb: Array.from(plot.targetDb),
  resultDb: Array.from(plot.resultDb),
}));
