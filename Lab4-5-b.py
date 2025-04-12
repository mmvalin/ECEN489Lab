import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_bits = 3
LSB = 1  # Assume normalized LSB
ideal_transitions = np.arange(1, 2**n_bits) * LSB  # Ideal transition levels

# Given DNL in LSB
DNL = np.array([0, -0.5, 0.5, -1, 0.5, 0.5, 0])

# Compute actual transitions (ideal + cumulative DNL)
actual_transitions = ideal_transitions + np.cumsum(DNL)

# Offset and full-scale error
offset_error = 0.5 * LSB
full_scale_error = 0.5 * LSB

# Apply offset error
actual_transitions += offset_error

# Apply gain error (full scale)
ideal_range = actual_transitions[-1] - actual_transitions[0]
scaled_range = ideal_range + full_scale_error
gain_correction = scaled_range / ideal_range
actual_transitions = actual_transitions[0] + (actual_transitions - actual_transitions[0]) * gain_correction

# Create input vs output transfer curve
vin = [0] + list(actual_transitions) + [actual_transitions[-1] + LSB]
codes = np.arange(0, 2**n_bits + 1)

# Plot transfer curve
plt.step(vin, codes, where='post')
plt.xlabel("Input Voltage (LSB)")
plt.ylabel("ADC Output Code")
plt.title("3-bit ADC Transfer Curve with DNL, Offset and Full-Scale Error")
plt.grid(True)
plt.show()
