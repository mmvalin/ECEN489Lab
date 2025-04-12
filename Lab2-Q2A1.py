import numpy as np
import matplotlib.pyplot as plt

f_sig = 200.1e6       # Signal frequency (200.1 MHz)
fs    = 401e6         # Sampling frequency (401 MHz > Nyquist ~ 400.2 MHz)
n_bits = 6            # Quantizer resolution: 6 bits
n_periods_list = [30, 100]  # Number of signal periods to sample
full_scale = 1.0      # Full-scale amplitude from -1 to +1
# ------------------------------------------------------

levels = 2**n_bits               # e.g. 64 levels for 6 bits
delta = 2 * full_scale / levels  # Quantization step size

for n_periods in n_periods_list:
    # Time-related computations
    T_sig = 1 / f_sig   # Period of the signal
    T_s   = 1 / fs      # Sampling period

    # Determine how many samples to capture ~n_periods of the sine wave
    N = int(np.round(n_periods * T_sig / T_s))
    t = np.arange(N) * T_s

    # Generate full-scale sine wave
    x = np.sin(2 * np.pi * f_sig * t)

    # Quantize: round to nearest step, then clip
    x_q = np.clip(
        np.round(x / delta) * delta,
        -full_scale,
        full_scale - delta
    )

    # Compute FFT -> PSD
    X    = np.fft.fft(x_q)
    PSD  = (np.abs(X)**2) / (N * fs)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Consider only positive frequencies
    pos_idx = np.where(freqs >= 0)
    freqs   = freqs[pos_idx]
    PSD     = PSD[pos_idx]

    # Identify bin closest to the fundamental frequency
    fund_idx = np.argmin(np.abs(freqs - f_sig))
    signal_power = PSD[fund_idx]
    noise_power  = np.sum(PSD) - signal_power
    SNR_dB = 10 * np.log10(signal_power / noise_power)

    print(f"n_periods={n_periods}, N={N} samples => SNR = {SNR_dB:.2f} dB")

    # Plot PSD in dB vs. frequency
    plt.figure(figsize=(7, 4))
    plt.plot(freqs/1e6, 10*np.log10(PSD), label="PSD")
    plt.axvline(f_sig/1e6, color="r", linestyle="--", label="Signal Freq")
    plt.title(f"PSD (n_periods={n_periods}, fs={fs/1e6} MHz)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.grid(True)
    plt.xlim(0, fs/2 / 1e6)
    plt.legend()
    plt.tight_layout()
    plt.show()
