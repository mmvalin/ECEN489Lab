import numpy as np
import matplotlib.pyplot as plt

# Given parameters
f_in = 1e9  # Input signal frequency (1 GHz)
f_s = 10e9  # Sampling frequency (10 GHz)
A = 2  # Amplitude (2V)
tau = 10e-12  # Time constant (10 ps)
T_s = 1 / f_s  # Sampling period
T_in = 1 / f_in  # Input signal period

# Time vector for one input period
t = np.linspace(0, 5 * T_in, 10000)

# Sine wave input signal
v_in = A * np.sin(2 * np.pi * f_in * t)

# Square wave sampling carrier
sampling_carrier = 0.5 * (1 + np.sign(np.sin(2 * np.pi * f_s * t)))

# Initialize output signal
v_out = np.zeros_like(t)
sampled = False

# Simulate sampling and holding
for i in range(len(t)):
    if sampling_carrier[i] == 1:
        # Charging capacitor considering time constant (RC charging)
        if i == 0:
            v_out[i] = v_in[i]  # Initial sample
        else:
            v_out[i] = v_out[i - 1] + (v_in[i] - v_out[i - 1]) * (1 - np.exp(-(t[i] - t[i - 1]) / tau))
    else:
        v_out[i] = v_out[i - 1]


# Plotting the signals
plt.figure(figsize=(10, 6))
plt.plot(t * 1e9, v_in, label='Sine Wave Input Signal (1 GHz)', linestyle='--')
plt.plot(t * 1e9, v_out, label='Sampled Output (ZOH)', color='red')
plt.title('Zero-Order Hold (ZOH) Sampling Circuit Output')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()