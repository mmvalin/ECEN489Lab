import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

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

# Compute PSD using Welch's method
f, Pxx = welch(noisy_signal, Fs, nperseg=1024)

# Compute SNR from PSD
signal_bin = np.argmax(Pxx)  # Peak frequency bin
P_signal_measured = Pxx[signal_bin]


# Exclude the signal bin from noise estimation
Pxx_excluding_signal = np.delete(Pxx, signal_bin)  # Remove the peak bin

# Compute noise power more accurately
P_noise_measured = np.mean(Pxx_excluding_signal)  # Average power of remaining frequencies

SNR_measured_dB = 10 * np.log10(P_signal_measured / P_noise_measured)

# Compute variance of uniformly distributed noise for the same SNR
var_uniform_noise = (12 * P_noise)  # Uniform noise variance = 12 * Gaussian noise variance

# Plot results
plt.figure(figsize=(10, 5))


plt.semilogy(f, Pxx)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")
plt.title("Power Spectral Density (PSD)")

plt.tight_layout()
plt.show()

# Print results
print(f"Theoretical Noise Variance (Gaussian): {P_noise:.6e}")
print(f"Measured SNR from PSD: {SNR_measured_dB:.2f} dB")
print(f"Variance of Uniformly Distributed Noise: {var_uniform_noise:.6e}")
