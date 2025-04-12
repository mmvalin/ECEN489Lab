import numpy as np
import matplotlib.pyplot as plt

def original_signal(t, f1=100e6):  # Define the original signal
    return np.cos(2 * np.pi * f1 * t)

# Sampling parameters
Fs = 1000e6  # Sampling frequency (800 MHz)
Ts = 1 / Fs  # Sampling period
F1 = 100e6  # Signal frequency (100 MHz)
T = 10 / F1  # Duration for 10 cycles

t_cont = np.linspace(0, T, 1000)  # Continuous time for smooth plotting
x_cont = original_signal(t_cont, F1)  # Original signal

# Sampled signal
t_samples = np.arange(0, T, Ts)  # Sample points
x_samples = original_signal(t_samples, F1)  # Sampled values

# Reconstruction using sinc interpolation
def sinc_interp(x, t_samples, t):
    return np.sum(x * np.sinc((t[:, None] - t_samples) / Ts), axis=1)

t_recon = np.linspace(0, T, 1000)  # Time for reconstruction
x_recon = sinc_interp(x_samples, t_samples, t_recon)

# Compute Mean Squared Error (MSE)
MSE = np.mean((x_recon - x_cont) ** 2)
print(f"Mean Squared Error (MSE): {MSE}")

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

axs[0].plot(t_cont, x_cont, 'b', label='Original Signal')
axs[0].set_title('Original Signal')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')
axs[0].legend()
axs[0].grid()

axs[1].stem(t_samples, x_samples, 'r', markerfmt='ro', basefmt=" ", linefmt='r-', label='Sampled Points')
axs[1].set_title('Sampled Signal')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid()

axs[2].plot(t_recon, x_recon, 'g', label='Reconstructed Signal')
axs[2].set_title('Reconstructed Signal')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Amplitude')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()