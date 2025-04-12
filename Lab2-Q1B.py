import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window
import pandas as pd

# Given parameters
Fs = 5e6  # Sampling frequency (Hz)
f0 = 2e6  # Signal frequency (Hz)
A = 1.0   # Amplitude of the sine wave
SNR_dB = 50  # Desired SNR in dB

# Generate time vector
T = 1e-3  # Signal duration (1ms)
t = np.arange(0, T, 1/Fs)

# Generate clean sine wave
signal = A * np.sin(2 * np.pi * f0 * t)

# Compute signal power
P_signal = np.mean(signal**2)

# Compute noise power for required SNR
SNR_linear = 10**(SNR_dB / 10)
P_noise = P_signal / SNR_linear
sigma_noise = np.sqrt(P_noise)

# Generate Gaussian noise
noise = np.random.normal(0, sigma_noise, size=t.shape)
noisy_signal = signal + noise

# Define windows
windows = {
    "Hanning": get_window("hann", 1024),
    "Hamming": get_window("hamming", 1024),
    "Blackman": get_window("blackman", 1024)
}

# Store results
snr_results = {}

# Compute PSD and SNR for each window
plt.figure(figsize=(10, 6))
for name, window in windows.items():
    f, Pxx = welch(noisy_signal, Fs, window=window, nperseg=1024)

    # Find signal bin
    signal_bin = np.argmax(Pxx)
    
    # Exclude signal bin from noise estimation
    bins_to_exclude = np.arange(signal_bin - 2, signal_bin + 3)
    bins_to_exclude = bins_to_exclude[(bins_to_exclude >= 0) & (bins_to_exclude < len(Pxx))]
    
    mask = np.ones(len(Pxx), dtype=bool)
    mask[bins_to_exclude] = False  # Exclude signal bins

    P_signal_measured = Pxx[signal_bin]
    P_noise_measured = np.mean(Pxx[mask])  # More accurate noise estimation

    SNR_measured_dB = 10 * np.log10(P_signal_measured / P_noise_measured)
    snr_results[name] = SNR_measured_dB

    # Plot PSD
    plt.semilogy(f, Pxx, label=name)
    

# Plot settings
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Power Spectral Density (PSD)")
plt.legend()
plt.show()


snr_df = pd.DataFrame.from_dict(snr_results, orient='index', columns=['Measured SNR (dB)'])
print("SNR Comparison for Different Windows:")
print(snr_df)