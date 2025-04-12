import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Given parameters
F = 2e6  # Signal frequency (Hz)
Fs = 5e6  # Sampling frequency (Hz)
N = 50  # Number of points in DFT

# Time vector
n = np.arange(N)
t = n / Fs  # Discrete time instances

# Generate the signal
x = np.cos(2 * np.pi * F * t)

# Compute DFT without windowing
X_no_window = fft(x, N)
X_magnitude_no_window = np.abs(X_no_window)

# Apply Blackman window
window = np.blackman(N)
x_windowed = x * window

# Compute DFT with Blackman window
X_windowed = fft(x_windowed, N)
X_magnitude_windowed = np.abs(X_windowed)

# Frequency axis
freqs = np.fft.fftfreq(N, d=1/Fs)

# Plot time-domain signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.plot(t, x_windowed, label='Blackman Windowed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Signals')
plt.legend()
plt.grid()

# Plot magnitude spectra
plt.subplot(2, 1, 2)
plt.plot(freqs[:N//2], X_magnitude_no_window[:N//2], 'r', label='Without Blackman')
plt.plot(freqs[:N//2], X_magnitude_windowed[:N//2], 'b', label='With Blackman')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
