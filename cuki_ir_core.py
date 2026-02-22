import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt


def oct_spectrum2(s, fs):
    """Compute 1/3-octave band spectrum of signal s sampled at fs."""
    b = 3
    dbref = 1
    G = 10 ** (3 / 10)
    fr = 1000
    x = np.arange(1, 43)
    Gstar = np.power(G, (x - 30) / b)
    fm = np.multiply(Gstar, fr)
    f2 = np.multiply(np.power(G, 1 / (2 * b)), fm)
    x  = np.delete(x,  np.where(f2 < 20),   axis=0)
    x  = np.delete(x,  np.where(f2 > fs/2), axis=0)
    fm = np.delete(fm, np.where(f2 < 20),   axis=0)
    fm = np.delete(fm, np.where(f2 > fs/2), axis=0)
    f2 = np.delete(f2, np.where(f2 < 20),   axis=0)
    f2 = np.delete(f2, np.where(f2 > fs/2), axis=0)
    f1 = np.multiply(np.power(G, -1 / (2 * b)), fm)

    S = np.zeros(len(x))
    for k in range(len(x)):
        B, A = signal.butter(2, [f1[k], f2[k]], btype='bandpass', fs=fs)
        sfilt = signal.lfilter(B, A, s)
        rms2b = np.sqrt(1 / len(sfilt) * sum(sfilt ** 2))
        S[k] = 10 * np.log10((rms2b / dbref) ** 2)

    rms2 = np.sqrt(1 / len(s) * sum(np.power(s, 2)))
    overall_lev  = 10 * np.log10(np.power(np.divide(rms2, dbref), 2))
    overall_levA = 10 * np.log10(np.sum(np.power(10, S / 10)))
    return S, fm, overall_lev, overall_levA, f1, f2


def save_ir_plot(ir, fs, NbF, title_waveform, title_spectrum, out_path):
    """Save a two-panel figure: IR waveform + log-frequency spectrum."""
    from matplotlib import pyplot as plt
    t    = np.arange(0, len(ir)) / fs
    FIRX = np.fft.fft(ir, NbF)
    freq = np.fft.fftfreq(len(t), t[1] - t[0])
    SdB  = 20 * np.log10(np.maximum(np.absolute(FIRX), 1e-12))  # avoid log(0)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(t * 1000, ir)
    axs[0].set_title(title_waveform)
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()

    axs[1].plot(freq[0:int(NbF / 2)], SdB[0:int(NbF / 2)])
    axs[1].set_title(title_spectrum)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("dB")
    axs[1].grid()
    axs[1].set_xscale('log')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  graph saved -> {out_path}")


def generate_irs(piezo_path: Path, mic_path: Path, ir_size: int, output_dir: Path, progress_callback=None) -> None:
    """Core DSP logic for generating Cuki IRs."""
    def report_progress(msg, pct=None):
        if progress_callback:
            progress_callback(msg, pct)

    if not piezo_path.exists():
        raise FileNotFoundError(f"Input file not found: {piezo_path}")
    if not mic_path.exists():
        raise FileNotFoundError(f"Input file not found: {mic_path}")

    NbF = ir_size
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "cuki"

    report_progress("Loading audio files...", 5)
    print("Loading audio files...")
    
    p_data, fs_p = sf.read(piezo_path)
    m_data, fs_m = sf.read(mic_path)
    
    if fs_p != fs_m:
        raise ValueError("Sample rates do not match.")
    fs = fs_p
    
    if p_data.ndim > 1: p_data = p_data[:, 0]
    if m_data.ndim > 1: m_data = m_data[:, 0]

    Nb = min(len(p_data), len(m_data))
    mic  = m_data[:Nb]
    pic  = p_data[:Nb]

    msg = f"Computing IR (NbF={NbF}) ..."
    report_progress(msg, 10)
    print(msg)
    FIR1  = np.zeros(NbF, dtype=complex)
    Nbuff = fs
    Nbmax = math.floor(Nb / Nbuff) - 10
    alice = np.zeros((NbF, Nbmax), dtype=complex)

    for n in range(Nbmax):
        i   = 3 * fs + n * Nbuff
        FIR = np.divide(
            np.fft.fft(mic[i:i + Nbuff - 1], NbF),
            np.fft.fft(pic[i:i + Nbuff - 1], NbF),
        )
        IR = np.real(np.fft.ifft(FIR, NbF))
        IR = IR / np.amax(np.absolute(IR))
        if any(np.isinf(FIR)) or any(np.isnan(FIR)):
            IR  = np.zeros(Nbuff)
            IR[0] = 1
            FIR = np.fft.fft(IR, NbF)
            print("  Warning: NaN or Inf detected in frame", n)
        alice[0:NbF, n] = FIR
        FIR1 = FIR1 + FIR

        if n % 5 == 0 or n == Nbmax - 1:
            report_progress(f"Extracting FFT from chunk {n+1}/{Nbmax}...", 10 + int((n/Nbmax)*20))

    ALICE = np.zeros(NbF, dtype=complex)
    for i in range(NbF):
        a = alice[i, :]
        A = a[np.absolute(np.absolute(a) - np.mean(a)) < 2 * np.std(a)]
        if any(np.isnan(A)) or any(np.isinf(A)):
            A = 1
            print(f"  Warning: NaN/Inf at frequency bin {i}")
        else:
            A = np.mean(A)
        ALICE[i] = A
        if i % 500 == 0 or i == NbF - 1:
            report_progress(f"Statistically averaging frequencies {i}/{NbF}...", 30 + int((i/NbF)*20))

    nn2       = np.arange(0, int(2 * NbF))
    window    = (.42 - .5 * np.cos(2 * np.pi * nn2 / (2 * NbF - 1))
                     + .08 * np.cos(4 * np.pi * nn2 / (2 * NbF - 1)))
    blackmanwin = window[NbF - 1:len(window) - 1]

    ir2  = np.fft.ifft(ALICE)
    ir2  = np.multiply(ir2, blackmanwin)
    ir2  = ir2 / np.amax(np.absolute(ir2)) * 0.95
    IR2  = np.real(ir2)

    nn3 = np.arange(10 * fs + 1, 20 * fs)
    MS  = mic[nn3]
    PS  = np.convolve(pic[nn3], IR2, 'same')
    p,  cf, _, _, f1, f2 = oct_spectrum2(MS / np.amax(np.absolute(MS)), fs)
    p2, _,  _, _, _,  _  = oct_spectrum2(PS / np.amax(np.absolute(PS)), fs)
    g0 = p - p2

    report_progress("Applying Graphic EQ matching...", 60)
    IRX = np.zeros(NbF); IRX[0] = 1
    IR1 = IR2.copy()
    for i in range(len(f1)):
        report_progress(f"Matching Graphic EQ Band {i+1}/{len(f1)}...", 60 + int((i/len(f1))*30))
        g  = 10 ** ((g0[i]) / 20)
        B, A = signal.butter(2, [f1[i], f2[i]], btype='bandpass', fs=fs)
        sfilt  = signal.lfilter(B, A, IRX) * (g - 1)
        IRX    = IRX + sfilt
        sfilt1 = signal.lfilter(B, A, IR1) * (g - 1)
        IR1    = IR1 + sfilt1

    IR1 = IR1 / np.amax(np.absolute(IR1)) * 0.95

    IRX = np.zeros(NbF); IRX[0] = 1
    IR3 = (IRX + IR2) / 2

    fmt = str(fs / 1000)
    prefix = f"IR_{stem}_{fmt[0:2]}k_{NbF}"

    for ir, suffix in [(IR1, "_M"), (IR2, "_Std"), (IR3, "_Std_Bld")]:
        out = output_dir / f"{prefix}{suffix}.wav"
        print(f"Saving {out} ...")
        sf.write(str(out), ir, fs, 'PCM_24')

    msg = "Generating plots ..."
    report_progress(msg, 95)
    print(msg)
    spec_dir = output_dir / "spectrum_graphs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    save_ir_plot(
        IR2, fs, NbF,
        title_waveform="IR Std",
        title_spectrum="Std Spectrum",
        out_path=spec_dir / f"{prefix}_Std.png",
    )
    save_ir_plot(
        IR1, fs, NbF,
        title_waveform="IR M-file",
        title_spectrum="M-file Spectrum",
        out_path=spec_dir / f"{prefix}_M.png",
    )

    # Generate comparison plot
    transformed_piezo = signal.convolve(p_data, IR1, mode='full')
    plot_nfft = min(8192, len(m_data))
    f_mic, Pxx_mic = signal.welch(m_data, fs, nperseg=plot_nfft)
    f_piezo, Pxx_piezo = signal.welch(transformed_piezo, fs, nperseg=plot_nfft)
    
    plt.figure(figsize=(12, 6))
    eps = 1e-12
    db_mic = 10 * np.log10(Pxx_mic + eps)
    db_piezo = 10 * np.log10(Pxx_piezo + eps)
    
    plt.semilogx(f_mic, db_mic, label='Target (Mic)', alpha=0.8, color='blue')
    plt.semilogx(f_piezo, db_piezo, label='Result (Pickup + IR M-file)', alpha=0.8, color='orange')
    
    plt.title('Frequency Response Comparison (Cuki Algorithm)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xlim(20, 20000)
    plt.ylim(max(np.min(db_mic), np.min(db_piezo)), max(np.max(db_mic), np.max(db_piezo)) + 10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plot_path = output_dir / "comparison_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Comparison plot saved to: {plot_path}")
