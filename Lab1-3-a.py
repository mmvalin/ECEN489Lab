import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Signal parameters
F1 = 800e6  # Frequency component 1 (200 MHz)
#F2 = 800e6  # Frequency component 2 (400 MHz)
Fs = 2000e6  # Sampling frequency (500 MHz)
N = 50      # Number of DFT points

# Time vector
t = np.arange(N) / Fs

# Sampled signal: sum of two cosines
y = np.cos(2 * np.pi * F1 * t)

# Compute the DFT using FFT
Y = fft(y, N)

# Frequency axis
freqs = np.fft.fftfreq(N, d=1/Fs)  # Compute frequency bins

# Plot the magnitude spectrum
plt.figure(figsize=(8, 4))
plt.stem(freqs[:N//2], np.abs(Y[:N//2]))  # Plot only positive frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude Spectrum of y(t)")
plt.grid()
plt.show()
