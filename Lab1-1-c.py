import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import dimpulse, lfilter

# Define the impulse signal (delta function)
N = 20  # Number of time samples
impulse = np.zeros(N)
impulse[0] = 1  # Unit impulse at n = 0

# Define the first system (FIR filter)
num1 = [1, 1, 1, 1, 1]  # FIR coefficients
den1 = [1]  # FIR filters have only a numerator

# Compute impulse response manually for FIR
h1 = lfilter(num1, den1, impulse)  # Impulse response of FIR
t1 = np.arange(len(h1))

# Define the second system (IIR filter)
num2 = [1, 1]
den2 = [1, -1]

# Compute impulse response using dimpulse for IIR
t2, h2 = dimpulse((num2, den2, 1), n=N)
t2, h2 = np.squeeze(t2), np.squeeze(h2)

# Plot the results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(t1, h1)
plt.title("Impulse Response of FIR Filter")
plt.xlabel("n (Time Steps)")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(t2, h2)
plt.title("Impulse Response of IIR Filter")
plt.xlabel("n (Time Steps)")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()
