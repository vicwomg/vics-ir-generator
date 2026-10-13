# Vic's Acoustic IR Generator

A robust DSP tool for generating acoustic guitar Impulse Responses (IRs). This tool takes the direct, sterile sound of an acoustic guitar's built-in piezo pickup and compares it to a high-quality studio microphone recording of the same performance. It then calculates and generates a mathematical Impulse Response (an `.wav` file) that can be loaded into IR pedals or plugins (like the HX Stomp, Nux Pulse, Sonicake IR, or TC Electronic Impulse Loader) to make your live piezo tone sound like a mic'd acoustic guitar.

If you find this useful, please consider supporting the project's ongoing development:

<a href="https://www.buymeacoffee.com/vicwomg" target="_blank"
          ><img
            src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png"
            alt="Buy Me A Coffee"
            style="height: 60px !important; width: 217px !important"
        /></a>

## How the Algorithm Works

The core algorithm is designed to match the frequency response of your pickup to your microphone, prioritizing musicality over perfect mathematical precision (which often sounds harsh and unnatural).

1. **Windowed FFT Extraction**: The algorithm slices both audio files into one-second chunks. For each chunk, it performs a Fast Fourier Transform (FFT) to extract the frequency spectrum and calculates the transfer function (Mic / Piezo).
2. **Statistical Outlier Rejection**: Acoustic instruments are dynamic. Fret buzz, string squeaks, or phase issues can ruin an IR. The algorithm statistically evaluates every frequency bin across all chunks, rejecting any anomalies that fall outside 2 standard deviations of the median spectrum.
3. **Fractional Octave Smoothing**: Raw FFT frequency matching creates a hyper-accurate but aggressive "comb filter" effect that sounds robotic. The algorithm heavily smoothes the peaks and valleys across fractional octaves (default: `1/3`) to retain the natural, wooden resonance of the instrument.
4. **Minimum Phase Conversion & High/Low Pass**: To ensure the IR feels instantaneous to play and doesn't cause latency, the matched frequency response is processed into a Minimum Phase impulse. Extreme sub-bass (< 60Hz) and harsh digital highs (> 18kHz) are rolled off to prevent speaker damage and hiss.
5. **Graphic EQ Matching**: As a final polish, the generated IR is mathematically convolved against the original piezo signal. A simulated 1/3-octave graphic EQ then fine-tunes the result, ensuring the final output matches the target microphone's energy footprint.

## Command Line Interface (CLI)

The Python CLI uses `uv` for dependency management. Install it by following the [`uv` installation guide](https://docs.astral.sh/uv/getting-started/installation/), then run the built-in CLI module from your terminal.

```bash
# Run the generator
uv run vics-ir-generator <piezo_path.wav> <mic_path.wav> [OPTIONS]
```

### CLI Arguments

- `piezo_file`: (Required) Path to the direct piezo recording WAV.
- `mic_file`: (Required) Path to the target microphone recording WAV.
- `--output`: Path to save the generated `.wav` file (default: `./output.wav`).
- `--ir-length`: Length of the final IR in samples. `2048` is the recommended default (approx. 46ms), offering a sweet spot of detail without excessive low-end ringing.
- `--smoothing`: Fractional octave smoothing applied to the IR. Higher matches are smoother but less detailed. Default is `0.3333` (1/3 octave).
- `--plot`: If included, generates and saves a PNG frequency response comparison graph alongside the output WAV.

Example:

```bash
uv run vics-ir-generator ./takes/guitar_direct.wav ./takes/guitar_mic.wav --output my_custom_ir.wav --plot
```

## Web Interface

<a href="https://ibb.co/LXJkzBDT"><img src="https://i.ibb.co/ks9HgzVL/vicsirgenerator.jpg" alt="vicsirgenerator" border="0"></a>

For ease of use, this project includes a static web interface. The frontend features drag-and-drop file selection, adjustable parameters, progress updates, response plots, and audio previews.

All DSP runs locally in a Web Worker. The selected recordings never leave the browser, and the site needs no Python service or server-side compute. Both the Standard and Cuki algorithms are supported: Standard downloads a 16-bit WAV, while Cuki downloads a ZIP containing its three 24-bit WAV variants. The page's Welch response plot compares the target microphone with the pickup after actual convolution with the generated IR.

The browser port follows the same operations as NumPy, SciPy, and libsndfile. Floating-point FFT implementations are not bit-stable across runtimes, so exact file hashes are not promised. On the provided SM57 sample pair at the default 2048-sample setting, the Standard output differs from the Python reference at 4 samples by one 16-bit least-significant bit. Each differing Cuki sample is likewise within one 24-bit least-significant bit.

There is a free-tier hosted version at: https://vics-ir-generator.onrender.com/ (it will probably take a long time to load and no guarantees on uptime or performance)

### Running via Docker

Build the Docker image:

```bash
docker build -t vics-ir-generator:latest .
```

Run the container mapping port 8000:

```bash
docker run -p 8000:80 vics-ir-generator:latest
```

Open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000) to use the UI

### Local Web Development

The browser uses JavaScript modules and a Web Worker, so serve the directory over HTTP rather than opening `index.html` directly:

```bash
python3 -m http.server 8000 --directory static
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Any static host (GitHub Pages, Cloudflare Pages, Netlify, S3, and similar) can deploy the contents of `static/` directly.

The FastAPI files remain in the repository as a Python reference implementation and for compatibility with existing deployments, but the current web page does not call the API.

### JavaScript DSP tests

Run the dependency-free unit tests with Node.js 20 or newer:

```bash
node --test tests/dsp.test.mjs
```
