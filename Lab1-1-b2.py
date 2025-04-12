import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Low-pass FIR Filter Coefficients 
FIR_num = [1, 1, 1, 1, 1]
FIR_den = [1]  # FIR filters have only numerator terms

# 2nd-order Butterworth Low-pass IIR Filter
IIR_num = [1, 1]
IIR_den = [1, -1]

# Compute zeros and poles
zeros_FIR, poles_FIR, _ = signal.tf2zpk(FIR_num, FIR_den)
zeros_IIR, poles_IIR, _ = signal.tf2zpk(IIR_num, IIR_den)

# Plot settings
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

def Show_PZ(ax, zeros, poles, title):
    ax.scatter(np.real(zeros), np.imag(zeros), s=100, marker='o', edgecolors='b', facecolors='none', label="Zeros")
    ax.scatter(np.real(poles), np.imag(poles), s=100, marker='x', color='r', label="Poles")
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.add_patch(plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed'))
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Plot FIR filter
Show_PZ(ax[0], zeros_FIR, poles_FIR, "FIR Filter (Pole-Zero Plot)")

# Plot IIR filter
Show_PZ(ax[1], zeros_IIR, poles_IIR, "IIR Filter (Pole-Zero Plot)")

# Show plot
plt.show()
