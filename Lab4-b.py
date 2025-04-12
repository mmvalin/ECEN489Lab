import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
bits = 12
Vfs = 1.2       # ADC full-scale peak-to-peak voltage [V]
Amplitude = 0.6     # RMS value of input signal [V]
fs = 10000          # Sampling frequency [Hz]
f = 100             # Input sinewave frequency [Hz]
duration = 1.0      # Duration of signal [s]

# Derived values
t = np.arange(0, duration, 1/fs)
N = len(t)
step_size = Vfs / (2**bits)

# Generate sinewave 
signal = Amplitude * np.sqrt(2) * np.sin(2 * np.pi * f * t)

# Create noise
noise_std = 1
noise_gauss = np.random.normal(0, noise_std, size=signal.shape)

# Add noise to signal
signal +=  noise_gauss

# Quantize the signal
quantized_signal = np.round(signal / step_size)


# FFT of noisy signal
yf = fft(signal)
xf = fftfreq(N, 1/fs)

# Use only positive frequencies
xf_pos = xf[:N//2]
yf_pos = np.abs(yf[:N//2])

# Power spectrum
power_spectrum = yf_pos**2
signal_bin = np.argmax(power_spectrum)
signal_power = power_spectrum[signal_bin]
noise_power = np.sum(power_spectrum) - signal_power

# SNR calculation (FFT-based)
snr_fft = 10 * np.log10(signal_power / noise_power)

# Display results
print(f"SNR Noisy Signal (FFT-based): {snr_fft:.2f} dB")

# Normalize power spectrum by signal power
normalized_psd = power_spectrum / signal_power

# Plot normalized power spectrum
plt.figure()
plt.semilogy(xf_pos, normalized_psd)
plt.title("Normalized Power Spectrum Noisy Signal (PSD / Signal Power)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized Power")
plt.grid(True)
plt.tight_layout()
plt.show()




# FFT of quantized signal
yf = fft(quantized_signal)
xf = fftfreq(N, 1/fs)

# Use only positive frequencies
xf_pos = xf[:N//2]
yf_pos = np.abs(yf[:N//2])

# Power spectrum
power_spectrum = yf_pos**2
signal_bin = np.argmax(power_spectrum)
signal_power = power_spectrum[signal_bin]
noise_power = np.sum(power_spectrum) - signal_power

# SNR calculation (FFT-based)
snr_fft = 10 * np.log10(signal_power / noise_power)

# Display results
print(f"SNR ADC output (FFT-based): {snr_fft:.2f} dB")

# Normalize power spectrum by signal power
normalized_psd = power_spectrum / signal_power

# Plot normalized power spectrum
plt.figure()
plt.semilogy(xf_pos, normalized_psd)
plt.title("Normalized Power Spectrum ADC output (PSD / Signal Power)")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized Power")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot original Signal
plt.figure()
plt.plot(t[:100], signal[:100], label='Original Signal')
plt.title('Signal (Zoomed)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot ADC output
plt.figure()
plt.plot(t[:100], quantized_signal[:100], label='ADC output', color='red', linestyle='--')
plt.title('ADC output (Zoomed)')
plt.xlabel('Time [s]')
plt.ylabel('ADC output')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
