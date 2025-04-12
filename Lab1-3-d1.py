import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Given parameters
F1 = 200e6  # First signal frequency (Hz)
F2 = 400e6  # Second signal frequency (Hz)
Fs = 1e9  # Sampling frequency (Hz)
N = 200  # Increased number of points in DFT for better resolution

# Time vector
n = np.arange(N)
t = n / Fs  # Discrete time instances

# Generate the signal
#y = np.cos(2 * np.pi * F1 * t)
y= np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# Compute DFT without windowing
Y_no_window = fft(y, N)
Y_magnitude_no_window = np.abs(Y_no_window)

# Apply Blackman window
window = np.blackman(N)
y_windowed = y * window

# Normalize the windowed signal to maintain amplitude consistency
y_windowed = y_windowed / np.max(window)

# Compute DFT with Blackman window
Y_windowed = fft(y_windowed, N)
Y_magnitude_windowed = np.abs(Y_windowed)

# Frequency axis
freqs = np.fft.fftfreq(N, d=1/Fs)

# Plot time-domain signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t[:100], y[:100], label='Original Signal')  # Plot only a subset for clarity
plt.plot(t[:100], y_windowed[:100], label='Windowed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time-Domain Signals')
plt.legend()
plt.grid()

# Plot magnitude spectra
plt.subplot(2, 1, 2)
plt.plot(freqs[:N//2], Y_magnitude_no_window[:N//2], 'r', label='Normal')
plt.plot(freqs[:N//2], Y_magnitude_windowed[:N//2], 'b', label='W/ Blackman')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()