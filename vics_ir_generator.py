import argparse
import numpy as np
import scipy.signal as signal
import soundfile as sf
import matplotlib.pyplot as plt

def oct_spectrum2(s, fs):
    """
    Calculates the 1/3 octave spectrum of a signal, returning the band levels 
    and center frequencies. Adapted from Cuki's IR generator.
    """
    b = 3
    dbref = 1
    
    L = len(s)
    G = 10**(3/10) # octave ratio, base 10
    fr = 1000 # reference frequency
    x = np.arange(1, 43) # band number
    Gstar = np.power(G, (x - 30)/b)
    fm = np.multiply(Gstar, fr) # midband frequency
    f2 = np.multiply(np.power(G, 1/(2*b)), fm) # upper bandedge freq.
    
    # Filter out bands outside of 20Hz - Nyquist
    valid_idx = (f2 >= 20) & (f2 <= fs/2)
    x = x[valid_idx]
    fm = fm[valid_idx]
    f2 = f2[valid_idx]
    f1 = np.multiply(np.power(G, -1/(2*b)), fm) # lower bandedge freq.
    
    S = np.zeros(len(x))
    
    # filtering
    for k in range(0, len(x)):
        # Calculate power in each 1/3 octave band
        B, A = signal.butter(2, [f1[k], f2[k]], btype='bandpass', fs=fs)
        sfilt = signal.lfilter(B, A, s)
        rms2b = np.sqrt(1/len(sfilt) * np.sum(sfilt**2))  
        # octave spectrum band level in dB
        S[k] = 10 * np.log10((rms2b/dbref)**2 + 1e-12) 
        
    return S, fm, f1, f2

def octave_smoothing(freq_response, fraction=1/3):
    """
    Smooths the magnitude of a frequency response using a fractional octave window.
    """
    # Work with magnitude only for smoothing
    mag = np.abs(freq_response)
    smoothed_mag = np.copy(mag)
    n = len(mag)
    
    for i in range(1, n):
        # Calculate window width based on frequency index (octave-based)
        # Higher frequencies get wider smoothing windows
        width = int(i * (2**(fraction/2) - 2**(-fraction/2)))
        
        if width > 1:
            start = max(0, i - width // 2)
            end = min(n, i + width // 2)
            smoothed_mag[i] = np.mean(mag[start:end])
            
    # Re-apply the original phase to the smoothed magnitude
    phase = np.angle(freq_response)
    return smoothed_mag * np.exp(1j * phase)

def to_minimum_phase(ir):
    n = len(ir)
    S = np.fft.fft(ir)
    log_amplitude = np.log(np.abs(S) + 1e-10) 
    real_cepstrum = np.fft.ifft(log_amplitude).real
    
    window = np.zeros(n)
    window[0] = 1
    window[1:n//2] = 2
    window[n//2] = 1
    
    min_phase_cepstrum = real_cepstrum * window
    min_phase_spectrum = np.exp(np.fft.fft(min_phase_cepstrum))
    return np.fft.ifft(min_phase_spectrum).real

def generate_guitar_ir(piezo_path, mic_path, output_path, ir_length=2048, smoothing=1/3, plot=False):
    # 1. Load and Pre-process
    p_data, fs = sf.read(piezo_path)
    m_data, _ = sf.read(mic_path)
    if len(p_data.shape) > 1: p_data = p_data[:, 0]
    if len(m_data.shape) > 1: m_data = m_data[:, 0]

    # 2. Transfer Function Estimation (cuki_ir style robust chunking)
    NbF = ir_length
    Nbuff = fs # 1-second chunks
    Nb = min(len(p_data), len(m_data))
    
    # Calculate how many 1-second chunks we can process (leaving a small 10-sample margin like cuki)
    import math
    Nbmax = math.floor(Nb / Nbuff) - 10
    
    if Nbmax <= 0:
        raise ValueError("Audio files are too short. Need at least 2 seconds of audio.")
        
    print(f"Processing {Nbmax} chunks for robust IR generation...")
    
    # Store the FFT transfers for each chunk
    alice = np.zeros((NbF, Nbmax), dtype=complex)
    
    for n in range(Nbmax):
        # cuki starts 3 seconds into the file. We'll start 1 second in to skip potential initial click
        i = fs + n * Nbuff 
        
        # Calculate Transfer function H = fft(mic) / fft(piezo) for this chunk
        mic_chunk_fft = np.fft.fft(m_data[i:i+Nbuff-1], NbF)
        pic_chunk_fft = np.fft.fft(p_data[i:i+Nbuff-1], NbF)
        
        # Add epsilon to prevent divide by zero
        epsilon = 1e-12
        FIR = np.divide(mic_chunk_fft, pic_chunk_fft + epsilon)
        
        # Check for NaN or Inf (bad chunks)
        if np.any(np.isinf(FIR)) or np.any(np.isnan(FIR)):
            # If bad chunk, provide neutral IR (essentially just a spike)
            FIR = np.ones(NbF, dtype=complex)
            
        alice[:, n] = FIR

    # 3. Statistical outlier rejection and averaging
    ALICE = np.zeros(NbF, dtype=complex)
    
    for i in range(NbF): # Browse each frequency bin across all chunks
        a = alice[i, :]
        # Only keep chunk values that are within 2 standard deviations of the mean
        # This rejects fret buzz, knocks, and phase misalignments
        std_val = np.std(a)
        if std_val > 0:
            mask = np.absolute(np.absolute(a) - np.mean(a)) < 2 * std_val
            A = a[mask]
        else:
            A = a
            
        if np.any(np.isnan(A)) or np.any(np.isinf(A)) or len(A) == 0:
            ALICE[i] = 1.0 # neutral fallback
        else:
            ALICE[i] = np.mean(A)

    # 3.5 Apply Fractional Octave Smoothing
    if smoothing > 0:
        ALICE = octave_smoothing(ALICE, fraction=smoothing)

    # 4. Generate Raw IR and apply Blackman Window
    ir_raw = np.fft.ifft(ALICE)
    
    nn2 = np.arange(0, int(2 * NbF))
    # Custom Blackman window matching cuki
    window = (.42 - .5 * np.cos(2 * np.pi * nn2 / (2 * NbF - 1)) + .08 * np.cos(4 * np.pi * nn2 / (2 * NbF - 1)))
    blackmanwin = window[NbF - 1:len(window) - 1]
    
    ir_raw = np.multiply(ir_raw, blackmanwin)
    ir_raw = np.real(ir_raw)
    # 5. Convert to Minimum Phase
    ir_min = to_minimum_phase(ir_raw)
    
    # 6. Final Polish (Pre-GEQ)
    ir_final = ir_min[:ir_length]
    fade_len = int(ir_length * 0.1)
    ir_final[-fade_len:] *= np.linspace(1, 0, fade_len)
    ir_final = ir_final / np.max(np.abs(ir_final)) * 0.95

    # 7. Graphic EQ Matching (The Cuki "M" Process)
    print("Applying Graphic EQ matching...")
    # Reference sample test at 10s and 20s (using middle of the file)
    test_start = min(10 * fs, len(m_data) // 3)
    test_end = min(20 * fs, len(m_data) // 3 * 2)
    
    MS = m_data[test_start:test_end]
    PS = signal.convolve(p_data[test_start:test_end], ir_final, mode='same')
    
    # Calculate 1/3 octave spectra
    p_mic, _, f1, f2 = oct_spectrum2(MS / (np.max(np.abs(MS)) + 1e-12), fs)
    p_piezo, _, _, _ = oct_spectrum2(PS / (np.max(np.abs(PS)) + 1e-12), fs)
    
    # Difference in dB between target mic and current piezo+IR result
    g0 = p_mic - p_piezo
    
    # Apply corrective EQ bands to the IR
    IR1 = np.copy(ir_final)
    IRX = np.zeros(ir_length)
    IRX[0] = 1.0 # identity impulse
    
    for i in range(0, len(f1)): # Browse each frequency of the octave spectrum
        g_db = g0[i]
        
        # Limit extreme EQ corrections to prevent runaway feedback/ringing
        g_db = np.clip(g_db, -24.0, 24.0)
            
        g_linear = 10**(g_db/20)
        
        # Isolate the frequency band
        B, A = signal.butter(2, [f1[i], f2[i]], btype='bandpass', fs=fs)
        
        # Apply the gain difference to that band and add it to the IR
        sfilt1 = signal.lfilter(B, A, IR1) * (g_linear - 1)
        IR1 = IR1 + sfilt1 
        
    # Final normalization
    ir_final = IR1 / np.max(np.abs(IR1)) * 0.99
    
    sf.write(output_path, ir_final.astype(np.float32), fs)
    print(f"Musical IR generated: {output_path} (Smoothing: {smoothing} octaves)")

    if plot:
        print("Applying IR and generating plot...")
        # Apply IR to piezo signal
        transformed_piezo = signal.convolve(p_data, ir_final, mode='full')
        
        # Calculate frequency response of both signals
        # We use a larger nperseg here for a highly detailed plot
        plot_nfft = min(8192, len(m_data)) 
        f_mic, Pxx_mic = signal.welch(m_data, fs, nperseg=plot_nfft)
        f_piezo, Pxx_piezo = signal.welch(transformed_piezo, fs, nperseg=plot_nfft)
        
        plt.figure(figsize=(12, 6))
        
        # Convert power spectra to dB for plotting
        # Adding epsilon to avoid log(0)
        eps = 1e-12
        db_mic = 10 * np.log10(Pxx_mic + eps)
        db_piezo = 10 * np.log10(Pxx_piezo + eps)
        
        # Offset the plots slightly so they are visually comparable but distinct
        plt.semilogx(f_mic, db_mic, label='Target (Mic)', alpha=0.8, color='blue')
        plt.semilogx(f_piezo, db_piezo, label='Result (Pickup + IR)', alpha=0.8, color='orange')
        
        plt.title('Frequency Response Comparison')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim(20, 20000) # Standard human hearing range
        plt.ylim(max(np.min(db_mic), np.min(db_piezo)), max(np.max(db_mic), np.max(db_piezo)) + 10)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        # Save plot alongside the output audio file
        plot_path = output_path.rsplit('.', 1)[0] + '.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        # We can also keep plt.show() if they still want the interactive window
        plt.show()
        plt.close('all') # Prevent memory leaks on the server

def main():
    parser = argparse.ArgumentParser(description="Generate a guitar IR from piezo and mic recordings.")
    parser.add_argument("piezo_path", help="Path to the piezo audio file")
    parser.add_argument("mic_path", help="Path to the mic audio file")
    parser.add_argument("--output_path", "-o", default="./output.wav", help="Output path (default: ./output.wav)")
    parser.add_argument("--ir_length", type=int, default=2048, help="Length of the generated IR (default: 2048)")
    parser.add_argument("--smoothing", type=float, default=1/3, help="Fractional octave smoothing (default: 1/3)")
    parser.add_argument("--plot", action="store_true", help="Plot a comparison of the target mic and generated piezo+IR frequency responses")
    
    args = parser.parse_args()
    
    generate_guitar_ir(
        args.piezo_path,
        args.mic_path,
        args.output_path,
        ir_length=args.ir_length,
        smoothing=args.smoothing,
        plot=args.plot
    )

if __name__ == "__main__":
    main()