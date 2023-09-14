"""
todo: Describe me
"""
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from tftb.generators import fmlin, amgauss
# https://tftb.readthedocs.io/en/latest/nonstationary_signals.html

# Stationary
fs = 32768
ts = np.linspace(0, 1, fs)
y1 = np.sin(2 * np.pi * 697 * ts)
y2 = np.sin(2 * np.pi * 1336 * ts)
sum = (y1 + y2) / 2
plt.figure()
plt.plot(ts, sum)
plt.xlim(0, 0.1)

y = sum[:int(fs / 16)]
y_analytic = hilbert(y)
plt.figure()
plt.plot(np.real(y_analytic), np.imag(y_analytic))
plt.xlabel("Real part")
plt.ylabel("Imaginary part")

# Nonstationary
y_nonstat, _ = fmlin(2048)  # Already analytic, no need of Hilbert transorm
y_nonstat *= amgauss(2048)
plt.figure()
plt.plot(np.real(y_nonstat), np.imag(y_nonstat))
plt.xlabel("Real part")
plt.ylabel("Imaginary part")

plt.show()
