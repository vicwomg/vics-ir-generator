import { createStoredZip, decodeWav, encodePcmWav, generateComparisonSpectrum, generateCukiIrs, generateStandardIr } from "./dsp.js";

self.onmessage = ({ data }) => {
  try {
    const piezo = decodeWav(data.piezo);
    const mic = decodeWav(data.mic);
    if (piezo.sampleRate !== mic.sampleRate) throw new Error("The input sample rates do not match.");
    const progress = (message, percent) => self.postMessage({ type: "progress", message, percent });

    if (data.algorithm === "cuki") {
      const result = generateCukiIrs(piezo.samples, mic.samples, piezo.sampleRate, data.irLength, progress);
      const rateLabel = String(piezo.sampleRate / 1000).slice(0, 2);
      const prefix = `IR_cuki_${rateLabel}k_${data.irLength}`;
      const matchedWav = encodePcmWav(result.matched, piezo.sampleRate, 24);
      const files = [
        { name: `${prefix}_M.wav`, data: matchedWav },
        { name: `${prefix}_Std.wav`, data: encodePcmWav(result.standard, piezo.sampleRate, 24) },
        { name: `${prefix}_Std_Bld.wav`, data: encodePcmWav(result.blended, piezo.sampleRate, 24) },
      ];
      const zip = createStoredZip(files);
      const plot = generateComparisonSpectrum(piezo.samples, mic.samples, result.matched, piezo.sampleRate, progress);
      self.postMessage({ type: "complete", algorithm: "cuki", download: zip, previewWav: matchedWav, plot, sampleRate: piezo.sampleRate }, [zip, matchedWav, plot.frequencies.buffer, plot.targetDb.buffer, plot.resultDb.buffer]);
    } else {
      const result = generateStandardIr(piezo.samples, mic.samples, piezo.sampleRate, data.irLength, data.smoothing, progress);
      const wav = encodePcmWav(result, piezo.sampleRate, 16);
      const plot = generateComparisonSpectrum(piezo.samples, mic.samples, result, piezo.sampleRate, progress);
      self.postMessage({ type: "complete", algorithm: "standard", download: wav, plot, sampleRate: piezo.sampleRate }, [wav, plot.frequencies.buffer, plot.targetDb.buffer, plot.resultDb.buffer]);
    }
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
