import numpy as np
import matplotlib.pyplot as plt

# Set parameters
fs = 400e6            # Sampling frequency: 400 MHz
n_bits = 12            # Quantizer resolution: 6 bits (64 levels)
f_sig = 200.1e6       # Signal frequency (using a slight offset to avoid exactly 2 samples per period)
full_scale = 1.0      # Full-scale amplitude (assumed to be from -1 to 1)
levels = 2**n_bits    # Total quantizer levels (64)
delta = 2 * full_scale / levels  # Quantization step size

# Test for both 30 and 100 periods (you can modify the list if needed)
for n_periods in [30, 100]:
    # Determine the number of samples needed
    T0 = 1 / f_sig              # Signal period (seconds)
    T = 1 / fs                  # Sampling period (seconds)
    N = int(np.round(n_periods * T0 / T))  # Total number of samples
    t = np.arange(N) * T        # Time vector

    # Generate a full-scale sine wave (phase = 0)
    x = np.sin(2 * np.pi * f_sig * t)

    # Quantize the sine wave:
    # Scale by delta, round to the nearest integer, then scale back.
    # Also, clip the values to the valid range of the quantizer.
    x_q = np.clip(np.round(x / delta) * delta, -full_scale, full_scale - delta)

    # No window is applied here (i.e. a rectangular window is implicit)
    # Compute the FFT and PSD of the quantized signal
    X = np.fft.fft(x_q)
    PSD = (np.abs(X)**2) / (N * fs)
    f_axis = np.fft.fftfreq(N, d=1/fs)

    # Consider only the positive frequencies.
    pos_idx = np.where(f_axis >= 0)
    f_axis = f_axis[pos_idx]
    PSD = PSD[pos_idx]

    # Identify the FFT bin corresponding to the fundamental frequency.
    fund_idx = np.argmin(np.abs(f_axis - f_sig))
    signal_power = PSD[fund_idx]
    noise_power = np.sum(PSD) - signal_power
    snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"Using {n_periods} periods (N = {N} samples): Estimated FFT SNR = {snr_db:.2f} dB")

    # Plot the PSD (in dB) versus frequency (MHz)
    plt.figure(figsize=(8, 4))
    plt.plot(f_axis / 1e6, 10 * np.log10(PSD))
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("PSD (dB)")
    plt.title(f"PSD ({n_periods} Periods)")
    plt.grid(True)
    plt.xlim(0, fs/2/1e6)  # Up to the Nyquist frequency (MHz)
    plt.show()
